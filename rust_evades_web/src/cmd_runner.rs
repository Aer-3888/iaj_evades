use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::broadcast;

#[derive(Clone, Debug, serde::Serialize)]
pub struct LogLine {
    pub stream: String,
    pub text: String,
}

pub struct CmdRunner {
    tx_logs: broadcast::Sender<LogLine>,
}

impl CmdRunner {
    pub fn new() -> (Self, broadcast::Receiver<LogLine>) {
        let (tx, rx) = broadcast::channel(1000);
        (Self { tx_logs: tx }, rx)
    }

    pub fn run(&self, cmd_str: String) {
        let tx = self.tx_logs.clone();
        tokio::spawn(async move {
            let mut child = Command::new("bash")
                .arg("-c")
                .arg(&cmd_str)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("failed to spawn command");

            let stdout = child.stdout.take().unwrap();
            let stderr = child.stderr.take().unwrap();

            let mut stdout_reader = BufReader::new(stdout).lines();
            let mut stderr_reader = BufReader::new(stderr).lines();

            let tx_out = tx.clone();
            let tx_err = tx.clone();

            let out_task = tokio::spawn(async move {
                while let Ok(Some(line)) = stdout_reader.next_line().await {
                    let _ = tx_out.send(LogLine {
                        stream: "stdout".to_string(),
                        text: line,
                    });
                }
            });

            let err_task = tokio::spawn(async move {
                while let Ok(Some(line)) = stderr_reader.next_line().await {
                    let _ = tx_err.send(LogLine {
                        stream: "stderr".to_string(),
                        text: line,
                    });
                }
            });

            let _ = child.wait().await;
            let _ = out_task.await;
            let _ = err_task.await;
            
            let _ = tx.send(LogLine {
                stream: "system".to_string(),
                text: "--- COMMAND FINISHED ---".to_string(),
            });
        });
    }
}
