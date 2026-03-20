import { useEffect, useRef, useState } from 'react';
import { useSocket } from '../contexts/SocketContext';
import { Maximize, Minimize } from 'lucide-react';

interface GameState {
  player: {
    body: {
      pos: { x: number; y: number };
      radius: number;
    };
  };
  enemies: Array<{
    body: {
      pos: { x: number; y: number };
      radius: number;
    };
  }>;
  config: {
    map_design: "Open" | "Closed";
    screen_width: number;
    screen_height: number;
    background_color: { r: number; g: number; b: number };
    player_color: { r: number; g: number; b: number };
    enemy_color: { r: number; g: number; b: number };
    grid_color: { r: number; g: number; b: number };
    grid_spacing: number;
    world_width: number;
    corridor_top: number;
    corridor_bottom: number;
    goal_width: number;
    start_margin: number;
    camera_lead: number;
    show_raycast: boolean;
    vision_only: boolean;
  };
  elapsed_time: number;
  enemies_evaded: number;
  base_seed: number;
  current_level: number;
  map: number[][];
  best_x: number;
  total_deaths: number;
  best_survival_ever: number;
  best_progress_ever: number;
}

interface Props {
  isRunning: boolean;
  isAiMode: boolean;
}

export default function Visualizer({ isRunning, isAiMode }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { sendMessage, subscribe } = useSocket();
  const [isFullscreen, setIsFullscreen] = useState(false);
  const lastStateRef = useRef<GameState | null>(null);
  const requestRef = useRef<number>();

  const pressedKeys = useRef<Set<string>>(new Set());

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);

    const unsubscribe = subscribe('Game', (data) => {
      lastStateRef.current = data;
    });

    const updateAction = () => {
      if (isAiMode) return; // Don't send manual actions in AI mode

      const up = pressedKeys.current.has('w') || pressedKeys.current.has('arrowup');
      const down = pressedKeys.current.has('s') || pressedKeys.current.has('arrowdown');
      const left = pressedKeys.current.has('a') || pressedKeys.current.has('arrowleft');
      const right = pressedKeys.current.has('d') || pressedKeys.current.has('arrowright');

      let action = 'Idle';
      if (up && right) action = 'UpRight';
      else if (up && left) action = 'UpLeft';
      else if (down && right) action = 'DownRight';
      else if (down && left) action = 'DownLeft';
      else if (up) action = 'Up';
      else if (down) action = 'Down';
      else if (left) action = 'Left';
      else if (right) action = 'Right';

      sendMessage({ type: 'Action', data: { action } });
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (['w', 's', 'a', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
        // Only prevent default if the game is running and we are in manual mode
        if (isRunning && !isAiMode) {
          e.preventDefault();
        }
        
        if (!pressedKeys.current.has(key)) {
          pressedKeys.current.add(key);
          updateAction();
        }
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (pressedKeys.current.has(key)) {
        pressedKeys.current.delete(key);
        updateAction();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    const animate = () => {
      draw();
      requestRef.current = requestAnimationFrame(animate);
    };
    requestRef.current = requestAnimationFrame(animate);

    return () => {
      unsubscribe();
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [sendMessage, subscribe, isAiMode, isRunning]);

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  const draw = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    const state = lastStateRef.current;

    if (!canvas || !ctx || !state) return;

    const { config } = state;
    const viewportWidth = config.screen_width || 1000;
    const viewportHeight = config.screen_height || 500;
    
    if (canvas.width !== viewportWidth || canvas.height !== viewportHeight) {
      canvas.width = viewportWidth;
      canvas.height = viewportHeight;
    }

    // Camera
    let camX, camY;
    if (config.map_design === 'Open') {
      camX = state.player.body.pos.x - viewportWidth / 2;
      camY = state.player.body.pos.y - viewportHeight / 2;
    } else {
      const target = state.player.body.pos.x - viewportWidth * 0.35 + (config.camera_lead || 140);
      camX = Math.max(0, Math.min(target, config.world_width - viewportWidth));
      camY = 0;
    }

    const pX = state.player.body.pos.x - camX;
    const pY = state.player.body.pos.y - camY;

    // Background
    ctx.fillStyle = `rgb(${config.background_color.r}, ${config.background_color.g}, ${config.background_color.b})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const drawWorld = () => {
      if (config.map_design === 'Open') {
        // Grid for Open Map
        ctx.strokeStyle = `rgb(${config.grid_color.r}, ${config.grid_color.g}, ${config.grid_color.b})`;
        ctx.lineWidth = 1;
        const spacing = config.grid_spacing || 64;
        const offsetX = -((camX % spacing + spacing) % spacing);
        const offsetY = -((camY % spacing + spacing) % spacing);

        ctx.beginPath();
        for (let x = offsetX; x < canvas.width; x += spacing) {
          ctx.moveTo(x, 0);
          ctx.lineTo(x, canvas.height);
        }
        for (let y = offsetY; y < canvas.height; y += spacing) {
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
        }
        ctx.stroke();
      } else {
        // Draw Closed map (corridor)
        const corridorHeight = config.corridor_bottom - config.corridor_top;
        ctx.fillStyle = 'rgba(40, 52, 68, 1)';
        ctx.fillRect(0, config.corridor_top - camY, canvas.width, corridorHeight);
        
        ctx.fillStyle = 'rgba(90, 112, 138, 1)';
        ctx.fillRect(0, config.corridor_top - camY, canvas.width, 3);
        ctx.fillRect(0, config.corridor_bottom - 3 - camY, canvas.width, 3);

        // Goal
        const goalX = config.world_width - config.goal_width;
        ctx.fillStyle = 'rgba(130, 218, 109, 0.5)';
        ctx.fillRect(goalX - camX, config.corridor_top - camY, config.goal_width, corridorHeight);
      }

      // Enemies
      ctx.fillStyle = `rgb(${config.enemy_color.r}, ${config.enemy_color.g}, ${config.enemy_color.b})`;
      for (const enemy of state.enemies) {
        const x = enemy.body.pos.x - camX;
        const y = enemy.body.pos.y - camY;
        ctx.beginPath();
        ctx.arc(x, y, enemy.body.radius, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    if (config.vision_only) {
      const RAY_COUNT = 36;
      const rayLength = state.player.body.radius * 5.0;
      const originX = state.player.body.pos.x;
      const originY = state.player.body.pos.y;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(pX, pY);

      for (let i = 0; i <= RAY_COUNT; i++) {
        const angle = (i % RAY_COUNT) * (360 / RAY_COUNT) * (Math.PI / 180);
        const dirX = Math.cos(angle);
        const dirY = -Math.sin(angle);

        let minDistance = Infinity;
        for (const enemy of state.enemies) {
          const dist = raycastCircleDistance(originX, originY, dirX, dirY, enemy.body.pos.x, enemy.body.pos.y, enemy.body.radius);
          if (dist !== null) minDistance = Math.min(minDistance, dist);
        }
        if (config.map_design === 'Closed') {
          const topDist = raycastHorizontalLineDistance(originY, dirY, config.corridor_top);
          if (topDist !== null) minDistance = Math.min(minDistance, topDist);
          const bottomDist = raycastHorizontalLineDistance(originY, dirY, config.corridor_bottom);
          if (bottomDist !== null) minDistance = Math.min(minDistance, bottomDist);
          const leftDist = raycastVerticalLineDistance(originX, dirX, 0);
          if (leftDist !== null) minDistance = Math.min(minDistance, leftDist);
          const rightDist = raycastVerticalLineDistance(originX, dirX, config.world_width);
          if (rightDist !== null) minDistance = Math.min(minDistance, rightDist);
        }

        const actualDistance = Math.min(minDistance, rayLength);
        const endX = (originX + dirX * actualDistance) - camX;
        const endY = (originY + dirY * actualDistance) - camY;
        ctx.lineTo(endX, endY);
      }
      ctx.closePath();
      ctx.clip();
      drawWorld();
      ctx.restore();
    } else {
      drawWorld();
    }

    // Player
    ctx.fillStyle = `rgb(${config.player_color.r}, ${config.player_color.g}, ${config.player_color.b})`;
    ctx.beginPath();
    ctx.arc(pX, pY, state.player.body.radius, 0, Math.PI * 2);
    ctx.fill();

    // Raycasting Visualization
    if (config.show_raycast) {
      const RAY_COUNT = 36;
      const nearRayLength = state.player.body.radius * 5.0;
      const farRayLength = nearRayLength * 2.0;
      const originX = state.player.body.pos.x;
      const originY = state.player.body.pos.y;

      ctx.save();
      ctx.lineWidth = 2;

      const hitByNear = new Set<number>();
      const nearRays: { actualDistance: number, endX: number, endY: number, minNear: number }[] = [];
      const farRays: { actualDistance: number, endX: number, endY: number, minFar: number }[] = [];

      // 1. Near Pass
      for (let i = 0; i < RAY_COUNT; i++) {
        const angle = i * (360 / RAY_COUNT) * (Math.PI / 180);
        const dirX = Math.cos(angle);
        const dirY = -Math.sin(angle);

        let minNear = Infinity;

        // Check enemies
        state.enemies.forEach((enemy, idx) => {
          const dist = raycastCircleDistance(
            originX, originY, dirX, dirY,
            enemy.body.pos.x, enemy.body.pos.y, enemy.body.radius
          );
          if (dist !== null && dist < minNear) {
            minNear = dist;
            hitByNear.add(idx);
          }
        });

        // Check walls in Closed mode
        if (config.map_design === 'Closed') {
          const topDist = raycastHorizontalLineDistance(originY, dirY, config.corridor_top);
          if (topDist !== null) minNear = Math.min(minNear, topDist);

          const bottomDist = raycastHorizontalLineDistance(originY, dirY, config.corridor_bottom);
          if (bottomDist !== null) minNear = Math.min(minNear, bottomDist);

          const leftDist = raycastVerticalLineDistance(originX, dirX, 0);
          if (leftDist !== null) minNear = Math.min(minNear, leftDist);

          const rightDist = raycastVerticalLineDistance(originX, dirX, config.world_width);
          if (rightDist !== null) minNear = Math.min(minNear, rightDist);
        }

        const actualDistance = Math.min(minNear, nearRayLength);
        const endX = (originX + dirX * actualDistance) - camX;
        const endY = (originY + dirY * actualDistance) - camY;

        nearRays.push({ actualDistance, endX, endY, minNear });
      }

      // 2. Far Pass
      for (let i = 0; i < RAY_COUNT; i++) {
        const angle = i * (360 / RAY_COUNT) * (Math.PI / 180);
        const dirX = Math.cos(angle);
        const dirY = -Math.sin(angle);

        let minFar = Infinity;

        // Check enemies (skip hitByNear)
        state.enemies.forEach((enemy, idx) => {
          if (hitByNear.has(idx)) return;
          const dist = raycastCircleDistance(
            originX, originY, dirX, dirY,
            enemy.body.pos.x, enemy.body.pos.y, enemy.body.radius
          );
          if (dist !== null && dist < minFar) {
            minFar = dist;
          }
        });

        // Check walls in Closed mode
        if (config.map_design === 'Closed') {
          const topDist = raycastHorizontalLineDistance(originY, dirY, config.corridor_top);
          if (topDist !== null) minFar = Math.min(minFar, topDist);

          const bottomDist = raycastHorizontalLineDistance(originY, dirY, config.corridor_bottom);
          if (bottomDist !== null) minFar = Math.min(minFar, bottomDist);

          const leftDist = raycastVerticalLineDistance(originX, dirX, 0);
          if (leftDist !== null) minFar = Math.min(minFar, leftDist);

          const rightDist = raycastVerticalLineDistance(originX, dirX, config.world_width);
          if (rightDist !== null) minFar = Math.min(minFar, rightDist);
        }

        const actualDistance = Math.min(minFar, farRayLength);
        const endX = (originX + dirX * actualDistance) - camX;
        const endY = (originY + dirY * actualDistance) - camY;

        farRays.push({ actualDistance, endX, endY, minFar });
      }

      // 3. Draw Far Rays (Cyan Blue)
      ctx.shadowBlur = 4;
      ctx.shadowColor = 'rgba(6, 182, 212, 0.4)';
      farRays.forEach(ray => {
        const alpha = 1.0 - (ray.actualDistance / farRayLength);
        ctx.strokeStyle = `rgba(6, 182, 212, ${0.2 + alpha * 0.5})`; // Cyan blue
        ctx.beginPath();
        ctx.moveTo(pX, pY);
        ctx.lineTo(ray.endX, ray.endY);
        ctx.stroke();

        if (ray.minFar <= farRayLength) {
          ctx.fillStyle = '#06b6d4'; // Cyan
          ctx.beginPath();
          ctx.arc(ray.endX, ray.endY, 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // 4. Draw Near Rays (Yellow)
      ctx.shadowColor = 'rgba(250, 204, 21, 0.4)';
      nearRays.forEach(ray => {
        const alpha = 1.0 - (ray.actualDistance / nearRayLength);
        ctx.strokeStyle = `rgba(250, 204, 21, ${0.3 + alpha * 0.7})`;
        ctx.beginPath();
        ctx.moveTo(pX, pY);
        ctx.lineTo(ray.endX, ray.endY);
        ctx.stroke();

        if (ray.minNear <= nearRayLength) {
          ctx.fillStyle = '#facc15';
          ctx.beginPath();
          ctx.arc(ray.endX, ray.endY, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      ctx.restore();
    }

    // Overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(15, 15, 180, 120);
    
    ctx.fillStyle = '#60a5fa';
    ctx.font = 'bold 16px monospace';
    if (config.map_design === 'Open') {
      ctx.fillText('DESIGN: OPEN', 25, 40);
    } else {
      const progress = state.player.body.pos.x - config.start_margin;
      const total = config.world_width - config.goal_width - config.start_margin;
      ctx.fillText(`PROGRESS: ${Math.floor(Math.max(0, progress))}/${Math.floor(total)}`, 25, 40);
    }
    
    ctx.font = '12px monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(`TIME: ${state.elapsed_time.toFixed(2)}s`, 25, 65);
    ctx.fillText(`EVADES: ${state.enemies_evaded}`, 25, 85);
    ctx.fillText(`BEST: ${config.map_design === "Open" ? state.best_survival_ever.toFixed(1) + "s" : Math.floor(state.best_progress_ever)}`, 25, 105);
    ctx.fillText(`DEATHS: ${state.total_deaths}`, 25, 125);
  };

  return (
    <div 
      ref={containerRef} 
      className={`relative group flex items-center justify-center bg-black overflow-hidden ${isFullscreen ? 'w-screen h-screen' : 'rounded-xl shadow-2xl border border-slate-800'}`}
    >
      <canvas 
        ref={canvasRef} 
        className="block"
        style={{ 
          width: isFullscreen ? '100%' : 'auto',
          height: isFullscreen ? '100%' : 'auto',
          maxWidth: '100%', 
          maxHeight: isFullscreen ? '100%' : 'calc(100vh - 250px)',
          objectFit: 'contain',
          aspectRatio: lastStateRef.current ? `${lastStateRef.current.config.screen_width} / ${lastStateRef.current.config.screen_height}` : '2/1'
        }}
      />
      
      <button
        onClick={toggleFullscreen}
        className="absolute top-4 right-4 p-2 bg-slate-900/50 hover:bg-slate-900 text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm border border-slate-700/50"
        title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
      >
        {isFullscreen ? <Minimize size={20} /> : <Maximize size={20} />}
      </button>
    </div>
  );
}

// Raycasting Utilities (Ported from Rust sensing.rs)
function raycastHorizontalLineDistance(originY: number, dirY: number, lineY: number): number | null {
  if (Math.abs(dirY) < 1e-6) return null;
  const t = (lineY - originY) / dirY;
  return t > 0 ? t : null;
}

function raycastVerticalLineDistance(originX: number, dirX: number, lineX: number): number | null {
  if (Math.abs(dirX) < 1e-6) return null;
  const t = (lineX - originX) / dirX;
  return t > 0 ? t : null;
}

function raycastCircleDistance(
  originX: number, originY: number,
  dirX: number, dirY: number,
  centerX: number, centerY: number,
  radius: number
): number | null {
  const offsetX = originX - centerX;
  const offsetY = originY - centerY;
  const projection = offsetX * dirX + offsetY * dirY;
  const c = offsetX * offsetX + offsetY * offsetY - radius * radius;

  if (c > 0 && projection > 0) return null;

  const discriminant = projection * projection - c;
  if (discriminant < 0) return null;

  let distance = -projection - Math.sqrt(discriminant);
  return distance < 0 ? 0 : distance;
}
