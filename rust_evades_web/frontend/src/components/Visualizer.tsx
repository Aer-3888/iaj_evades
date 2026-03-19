import { useEffect, useRef } from 'react';
import { useSocket } from '../contexts/SocketContext';

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
  const { sendMessage, subscribe } = useSocket();
  const lastStateRef = useRef<GameState | null>(null);
  const requestRef = useRef<number>();

  const pressedKeys = useRef<Set<string>>(new Set());

  useEffect(() => {
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
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [sendMessage, subscribe]);

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

    // Background
    ctx.fillStyle = `rgb(${config.background_color.r}, ${config.background_color.g}, ${config.background_color.b})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

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

    // Player
    ctx.fillStyle = `rgb(${config.player_color.r}, ${config.player_color.g}, ${config.player_color.b})`;
    const pX = state.player.body.pos.x - camX;
    const pY = state.player.body.pos.y - camY;
    ctx.beginPath();
    ctx.arc(pX, pY, state.player.body.radius, 0, Math.PI * 2);
    ctx.fill();

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
    <canvas 
      ref={canvasRef} 
      className="bg-black shadow-inner block mx-auto"
      style={{ 
        maxWidth: '100%', 
        height: 'auto', 
        maxHeight: 'calc(100vh - 250px)',
        aspectRatio: lastStateRef.current ? `${lastStateRef.current.config.screen_width} / ${lastStateRef.current.config.screen_height}` : '2/1'
      }}
    />
  );
}
