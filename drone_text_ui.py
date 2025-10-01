#!/usr/bin/env python3
"""
Text-based Web Drone Simulator
No graphics/rendering - pure text interface for WSL compatibility
"""

import mujoco
import numpy as np
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any


class TextDroneSimulator:
    def __init__(self, model_path: str = "drone_model.xml"):
        """Initialize text-only drone simulator."""
        print("📟 Initializing Text-based Drone Simulator...")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Control and state
        self.control_input = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "thrust": 0.0}
        self.thrust_base = 2.5
        self.thrust_max = 5.0
        self.control_gains = {"roll": 0.5, "pitch": 0.5, "yaw": 0.3, "thrust": 1.0}

        # Simulation state
        self.running = True
        self.drone_state = {}
        self.step_count = 0

        # Initialize drone
        self.reset_drone()

        print("✅ Text simulator ready!")

    def reset_drone(self):
        """Reset drone to initial position."""
        self.data.qpos[0:3] = [0.0, 0.0, 2.0]  # Position
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Quaternion
        self.data.qvel[:] = 0.0  # Velocities
        mujoco.mj_forward(self.model, self.data)
        print("🔄 Drone reset to starting position")

    def apply_control(self):
        """Apply control inputs to drone."""
        base_thrust = self.thrust_base
        thrust = base_thrust + self.control_input["thrust"] * self.control_gains["thrust"]
        roll = self.control_input["roll"] * self.control_gains["roll"]
        pitch = self.control_input["pitch"] * self.control_gains["pitch"]
        yaw = self.control_input["yaw"] * self.control_gains["yaw"]

        # Motor mixing
        motor1 = thrust + roll + pitch + yaw
        motor2 = thrust - roll + pitch - yaw
        motor3 = thrust - roll - pitch + yaw
        motor4 = thrust + roll - pitch - yaw

        motors = [max(0, min(motor, self.thrust_max)) for motor in [motor1, motor2, motor3, motor4]]
        self.data.ctrl[:] = motors

    def get_drone_state(self) -> Dict[str, Any]:
        """Get current drone state."""
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]

        # Calculate Euler angles from quaternion for display
        w, x, y, z = quat
        roll_rad = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch_rad = np.arcsin(2*(w*y - z*x))
        yaw_rad = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        return {
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "velocity": {"x": float(vel[0]), "y": float(vel[1]), "z": float(vel[2])},
            "angular_velocity": {"x": float(ang_vel[0]), "y": float(ang_vel[1]), "z": float(ang_vel[2])},
            "orientation": {
                "roll": float(np.degrees(roll_rad)),
                "pitch": float(np.degrees(pitch_rad)),
                "yaw": float(np.degrees(yaw_rad))
            },
            "quaternion": {"w": float(w), "x": float(x), "y": float(y), "z": float(z)},
            "time": float(self.data.time),
            "step": self.step_count,
            "controls": self.control_input.copy(),
            "motors": [float(x) for x in self.data.ctrl[:4]]
        }

    def simulation_loop(self):
        """Main simulation loop."""
        print("🚀 Starting simulation loop...")
        while self.running:
            # Apply controls and step simulation
            self.apply_control()
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1

            # Update drone state
            self.drone_state = self.get_drone_state()

            # Print telemetry occasionally
            if self.step_count % 200 == 0:
                pos = self.data.qpos[:3]
                print(f"Step {self.step_count}: Pos({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | Time: {self.data.time:.1f}s")

            time.sleep(0.01)  # 100 FPS simulation

    def start_simulation(self):
        """Start simulation in background thread."""
        sim_thread = threading.Thread(target=self.simulation_loop)
        sim_thread.daemon = True
        sim_thread.start()


class TextWebHandler(BaseHTTPRequestHandler):
    simulator = None

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self.serve_main_page()
        elif parsed_path.path == '/state':
            self.serve_state()
        elif parsed_path.path == '/control':
            self.handle_control(parsed_path.query)
        else:
            self.send_error(404)

    def serve_main_page(self):
        """Serve the main web interface."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>🚁 Text-based Drone Simulator</title>
    <style>
        body { font-family: 'Courier New', monospace; margin: 20px; background: #0a0a0a; color: #00ff00; }
        .container { max-width: 1200px; margin: 0 auto; }
        .panel { background: #1a1a1a; border: 1px solid #00ff00; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .telemetry { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        button {
            padding: 15px 20px; margin: 5px; font-size: 16px; font-family: 'Courier New', monospace;
            background: #003300; color: #00ff00; border: 1px solid #00ff00;
            border-radius: 3px; cursor: pointer; min-width: 120px;
        }
        button:hover { background: #004400; }
        button:active { background: #006600; }
        .value { color: #00ffff; font-weight: bold; }
        .ascii-art { font-size: 12px; line-height: 1.2; color: #ffff00; }
        .status { color: #ff8800; }
        .motor { color: #ff00ff; }
        h1, h2, h3 { color: #ffff00; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        pre { color: #00ff00; background: #000; padding: 10px; border: 1px solid #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚁 MuJoCo Text-Mode Drone Simulator</h1>
        <div class="status">WSL-Compatible | No Graphics Required | Pure Physics</div>

        <div class="panel">
            <h2>📊 Telemetry Dashboard</h2>
            <div class="telemetry">
                <div>
                    <h3>📍 Position & Velocity</h3>
                    <div>X: <span class="value" id="pos_x">0.00</span> m | Vx: <span class="value" id="vel_x">0.00</span> m/s</div>
                    <div>Y: <span class="value" id="pos_y">0.00</span> m | Vy: <span class="value" id="vel_y">0.00</span> m/s</div>
                    <div>Z: <span class="value" id="pos_z">2.00</span> m | Vz: <span class="value" id="vel_z">0.00</span> m/s</div>
                </div>
                <div>
                    <h3>🧭 Orientation</h3>
                    <div>Roll: <span class="value" id="roll">0.0</span>°</div>
                    <div>Pitch: <span class="value" id="pitch">0.0</span>°</div>
                    <div>Yaw: <span class="value" id="yaw">0.0</span>°</div>
                </div>
            </div>
            <div class="telemetry">
                <div>
                    <h3>⚡ Motor Outputs</h3>
                    <div>Motor 1: <span class="motor" id="motor1">2.5</span></div>
                    <div>Motor 2: <span class="motor" id="motor2">2.5</span></div>
                    <div>Motor 3: <span class="motor" id="motor3">2.5</span></div>
                    <div>Motor 4: <span class="motor" id="motor4">2.5</span></div>
                </div>
                <div>
                    <h3>⏱️ System Status</h3>
                    <div>Time: <span class="value" id="time">0.0</span>s</div>
                    <div>Steps: <span class="value" id="steps">0</span></div>
                    <div>FPS: <span class="value" id="fps">--</span></div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>🎮 Flight Controls</h2>
            <div class="controls">
                <div>
                    <h3>🎯 Translation</h3>
                    <div style="text-align: center;">
                        <div><button onmousedown="startControl('pitch', -0.5)" onmouseup="stopControl('pitch')">⬆️ FORWARD</button></div>
                        <div>
                            <button onmousedown="startControl('roll', -0.5)" onmouseup="stopControl('roll')">⬅️ LEFT</button>
                            <button onmousedown="startControl('roll', 0.5)" onmouseup="stopControl('roll')">➡️ RIGHT</button>
                        </div>
                        <div><button onmousedown="startControl('pitch', 0.5)" onmouseup="stopControl('pitch')">⬇️ BACK</button></div>
                    </div>
                </div>

                <div>
                    <h3>🔄 Rotation & Altitude</h3>
                    <div style="text-align: center;">
                        <div><button onmousedown="startControl('thrust', 0.5)" onmouseup="stopControl('thrust')">🔺 UP</button></div>
                        <div>
                            <button onmousedown="startControl('yaw', -0.5)" onmouseup="stopControl('yaw')">↶ YAW L</button>
                            <button onmousedown="startControl('yaw', 0.5)" onmouseup="stopControl('yaw')">↷ YAW R</button>
                        </div>
                        <div><button onmousedown="startControl('thrust', -0.5)" onmouseup="stopControl('thrust')">🔻 DOWN</button></div>
                    </div>
                </div>

                <div>
                    <h3>🛠️ System</h3>
                    <button onclick="hover()">⏸️ HOVER</button>
                    <button onclick="reset()">🔄 RESET</button>
                    <button onclick="emergency()">🚨 EMERGENCY</button>
                    <div style="margin-top: 15px;">
                        <div>Current Inputs:</div>
                        <div>R: <span class="value" id="ctrl_roll">0.0</span></div>
                        <div>P: <span class="value" id="ctrl_pitch">0.0</span></div>
                        <div>Y: <span class="value" id="ctrl_yaw">0.0</span></div>
                        <div>T: <span class="value" id="ctrl_thrust">0.0</span></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>📈 ASCII Flight Visualization</h2>
            <pre id="ascii_display" class="ascii-art">
                Initializing...
            </pre>
        </div>

        <div class="panel">
            <h3>⌨️ Keyboard Controls</h3>
            <div class="grid">
                <div><strong>W/S:</strong> Forward/Back</div>
                <div><strong>A/D:</strong> Left/Right</div>
                <div><strong>Q/E:</strong> Yaw Left/Right</div>
                <div><strong>R/F:</strong> Up/Down</div>
            </div>
        </div>
    </div>

    <script>
        let activeControls = {};
        let lastUpdateTime = Date.now();
        let updateCount = 0;

        // Update telemetry
        setInterval(() => {
            fetch('/state')
                .then(response => response.json())
                .then(data => {
                    // Position & Velocity
                    document.getElementById('pos_x').textContent = data.position.x.toFixed(2);
                    document.getElementById('pos_y').textContent = data.position.y.toFixed(2);
                    document.getElementById('pos_z').textContent = data.position.z.toFixed(2);
                    document.getElementById('vel_x').textContent = data.velocity.x.toFixed(2);
                    document.getElementById('vel_y').textContent = data.velocity.y.toFixed(2);
                    document.getElementById('vel_z').textContent = data.velocity.z.toFixed(2);

                    // Orientation
                    document.getElementById('roll').textContent = data.orientation.roll.toFixed(1);
                    document.getElementById('pitch').textContent = data.orientation.pitch.toFixed(1);
                    document.getElementById('yaw').textContent = data.orientation.yaw.toFixed(1);

                    // Motors
                    document.getElementById('motor1').textContent = data.motors[0].toFixed(2);
                    document.getElementById('motor2').textContent = data.motors[1].toFixed(2);
                    document.getElementById('motor3').textContent = data.motors[2].toFixed(2);
                    document.getElementById('motor4').textContent = data.motors[3].toFixed(2);

                    // System
                    document.getElementById('time').textContent = data.time.toFixed(1);
                    document.getElementById('steps').textContent = data.step;

                    // Controls
                    document.getElementById('ctrl_roll').textContent = data.controls.roll.toFixed(1);
                    document.getElementById('ctrl_pitch').textContent = data.controls.pitch.toFixed(1);
                    document.getElementById('ctrl_yaw').textContent = data.controls.yaw.toFixed(1);
                    document.getElementById('ctrl_thrust').textContent = data.controls.thrust.toFixed(1);

                    // FPS calculation
                    updateCount++;
                    if (updateCount % 10 === 0) {
                        const now = Date.now();
                        const fps = 10000 / (now - lastUpdateTime);
                        document.getElementById('fps').textContent = fps.toFixed(1);
                        lastUpdateTime = now;
                    }

                    // ASCII visualization
                    updateASCIIDisplay(data);
                });
        }, 100);

        function updateASCIIDisplay(data) {
            const x = data.position.x;
            const y = data.position.y;
            const z = data.position.z;
            const yaw = data.orientation.yaw;

            let display = `
    Flight Status: ${z > 0.5 ? 'AIRBORNE' : 'LANDED'}    Altitude: ${z.toFixed(1)}m

    Top-down View (Grid: 1m squares):

    +-------+-------+-------+-------+-------+
    |       |       |   ^   |       |       |
    |   -2  |   -1  |   0   |   +1  |   +2  | +2
    |       |       |   |   |       |       |
    +-------+-------+-------+-------+-------+
    |       |       |       |       |       |
    |       |       |       |       |       | +1
    |       |       |       |       |       |
    +-------+-------+${x >= -0.5 && x <= 0.5 && y >= -0.5 && y <= 0.5 ? '---🚁---' : '-------'}+-------+-------+
    |       |       |       |       |       |
    |       |       |       |       |       | 0
    |       |       |       |       |       |
    +-------+-------+-------+-------+-------+
    |       |       |       |       |       |
    |       |       |       |       |       | -1
    |       |       |       |       |       |
    +-------+-------+-------+-------+-------+
    |       |       |       |       |       |
    |       |       |       |       |       | -2
    |       |       |   |   |       |       |
    +-------+-------+-------+-------+-------+
                            v

    Position: (${x.toFixed(1)}, ${y.toFixed(1)})    Heading: ${yaw.toFixed(0)}°
            `;

            document.getElementById('ascii_display').textContent = display;
        }

        function startControl(type, value) {
            activeControls[type] = value;
            sendControl();
        }

        function stopControl(type) {
            activeControls[type] = 0;
            sendControl();
        }

        function sendControl() {
            const params = new URLSearchParams(activeControls);
            fetch('/control?' + params);
        }

        function hover() {
            activeControls = {};
            sendControl();
        }

        function reset() {
            fetch('/control?reset=true');
        }

        function emergency() {
            activeControls = {roll: 0, pitch: 0, yaw: 0, thrust: -1};
            sendControl();
            setTimeout(() => {
                activeControls = {};
                sendControl();
            }, 1000);
        }

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.repeat) return;
            switch(e.key.toLowerCase()) {
                case 'w': startControl('pitch', -0.5); break;
                case 's': startControl('pitch', 0.5); break;
                case 'a': startControl('roll', -0.5); break;
                case 'd': startControl('roll', 0.5); break;
                case 'q': startControl('yaw', -0.5); break;
                case 'e': startControl('yaw', 0.5); break;
                case 'r': startControl('thrust', 0.5); break;
                case 'f': startControl('thrust', -0.5); break;
            }
        });

        document.addEventListener('keyup', (e) => {
            switch(e.key.toLowerCase()) {
                case 'w': case 's': stopControl('pitch'); break;
                case 'a': case 'd': stopControl('roll'); break;
                case 'q': case 'e': stopControl('yaw'); break;
                case 'r': case 'f': stopControl('thrust'); break;
            }
        });
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_state(self):
        """Serve drone state as JSON."""
        if self.simulator:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.simulator.drone_state).encode())
        else:
            self.send_error(503)

    def handle_control(self, query_string):
        """Handle control commands."""
        if not self.simulator:
            self.send_error(503)
            return

        params = parse_qs(query_string)

        # Update control inputs
        for key in ['roll', 'pitch', 'yaw', 'thrust']:
            if key in params:
                self.simulator.control_input[key] = float(params[key][0])

        # Handle reset
        if 'reset' in params:
            self.simulator.reset_drone()

        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Start the text-based drone simulator."""
    print("📟 Starting Text-based Web Drone Simulator...")

    try:
        # Initialize simulator
        simulator = TextDroneSimulator()
        TextWebHandler.simulator = simulator

        # Start simulation
        simulator.start_simulation()

        # Start web server
        port = 8080
        with HTTPServer(('localhost', port), TextWebHandler) as httpd:
            print(f"✅ Text interface ready at: http://localhost:{port}")
            print("🌐 Open this URL in your browser for full telemetry!")
            print("📊 Complete flight data, ASCII visualization, real-time controls")
            print("⌨️  Keyboard: WASD + QE + RF controls")
            print("🖱️  Mouse: Click and hold buttons")
            print("Press Ctrl+C to stop...")
            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if 'simulator' in locals():
            simulator.running = False
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()