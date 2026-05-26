import os
import numpy as np
import time

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

# Import traffic signs and lights
from qvl.crosswalk import QLabsCrosswalk
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.yield_sign import QLabsYieldSign
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign

def spawn_traffic_elements(qlabs):
    """Spawn all traffic lights, signs, and crosswalks - EXACT POSITIONS FROM CODE 2"""
    
    print("\n🚦 Spawning traffic elements...")
    
    # 🚶 CROSSWALKS (4 total) - CODE 2 POSITIONS
    walks = []
    for i in range(4):
        walks.append(QLabsCrosswalk(qlabs))
    
    walks[0].spawn(location=[-5, 9.5, 0],
                    rotation=[0, 0, np.pi/2], 
                    scale=[1, 1, 0.75],
                    configuration=0)
    walks[1].spawn(location=[1.3, 16, 0],
                    rotation=[0, 0, 0], 
                    scale=[1, 1, 0.75],
                    configuration=0)
    walks[2].spawn(location=[7.7, 9.5, 0],
                    rotation=[0, 0, np.pi/2], 
                    scale=[1, 1, 0.75],
                    configuration=0)
    walks[3].spawn(location=[1.3, 3, 0],
                    rotation=[0, 0, 0], 
                    scale=[1, 1, 0.75],
                    configuration=0)
    
    # 🚦 TRAFFIC LIGHTS (4 total) - CODE 2 POSITIONS WITH ALTERNATING COLORS
    lights = []
    light_states = [0, 1, 2, 0]  # RED, YELLOW, GREEN, RED - All different!
    
    # Using CODE 2 exact positions and rotations
    lights.append(QLabsTrafficLight(qlabs))
    lights[0].spawn(location=[-22.313, 36.363, 0.0],
                    rotation=[0, 0, 135],  # Code 2 uses degrees directly
                    configuration=light_states[0])
    
    lights.append(QLabsTrafficLight(qlabs))
    lights[1].spawn(location=[-2.95, 5.6, 0],
                    rotation=[0, 0, 300],
                    configuration=light_states[1])
    
    lights.append(QLabsTrafficLight(qlabs))
    lights[2].spawn(location=[6.7, 5.7, 0],
                    rotation=[0, 0, -np.pi/2],
                    configuration=light_states[2])
    
    lights.append(QLabsTrafficLight(qlabs))
    lights[3].spawn(location=[24.387, 4.74, 0.2],
                    rotation=[0, 0, 0],
                    configuration=light_states[3])
    
    # ⚠️ YIELD SIGN (1 total) - CODE 2 POSITION
    yieldSign = QLabsYieldSign(qlabs)
    yieldSign.spawn(location=[0.4, -13, 0],
                    rotation=[0, 0, np.pi])
    
    # 🔄 ROUNDABOUT SIGNS (3 total) - CODE 2 POSITIONS
    roundAboutSigns = []
    for i in range(3):
        roundAboutSigns.append(QLabsRoundaboutSign(qlabs))
    
    roundAboutSigns[0].spawn(location=[24.5, 33, 0],
                            rotation=[0, 0, -np.pi/2])
    roundAboutSigns[1].spawn(location=[4.5, 40, 0],
                            rotation=[0, 0, np.pi])
    roundAboutSigns[2].spawn(location=[10.6, 28.5, 0],
                            rotation=[0, 0, np.pi])
    
    # 🛑 STOP SIGN (1 total) - CODE 2 POSITION
    stop = QLabsStopSign(qlabs)
    stop.spawn(location=[-0.508, -7.327, 0.2], 
               rotation=[0, 0, np.pi/2],
               scale=[1, 1, 1], 
               configuration=0, 
               waitForConfirmation=True)
    
    # Print summary
    state_names = ['RED', 'YELLOW', 'GREEN']
    states_display = [state_names[s] for s in light_states]
    print("   ✅ Crosswalks: 4")
    print(f"   ✅ Traffic Lights: 4 (States: {states_display})")
    print("   ✅ Yield Sign: 1")
    print("   ✅ Roundabout Signs: 3")
    print("   ✅ Stop Sign: 1")
    print("✅ All traffic elements spawned!\n")
    
    return {
        'lights': lights,
        'light_states': light_states,
        'walks': walks,
        'roundAboutSigns': roundAboutSigns,
        'yieldSign': yieldSign,
        'stop': stop
    }

def cycle_traffic_lights(traffic_elements, cycle_interval=8.0):
    """Cycle traffic lights through RED → YELLOW → GREEN → RED"""
    lights = traffic_elements['lights']
    light_states = traffic_elements['light_states']
    
    last_cycle_time = time.time()
    
    print("🚦 Traffic light cycling started (every 8 seconds)")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            current_time = time.time()
            
            # Check if it's time to cycle
            if current_time - last_cycle_time >= cycle_interval:
                # Cycle each light through states (0→1→2→0)
                for i in range(len(lights)):
                    light_states[i] = (light_states[i] + 1) % 3
                    lights[i].set_state(light_states[i])
                
                # Display current states
                state_names = ['RED', 'YELLOW', 'GREEN']
                states_display = [state_names[s] for s in light_states]
                print(f"🚦 LIGHTS CYCLED: {states_display}")
                
                last_cycle_time = current_time
            
            time.sleep(0.1)  # Small delay to prevent CPU spinning
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopping traffic light cycling...")

def setup(
        initialPosition=[22.564, -4.416, 0.005],
        initialOrientation=[0, 0, -300],
        rtModel=rtmodels.QCAR,
        spawnTrafficElements=True
    ):

    # Try to connect to Qlabs
    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    try:
        qlabs.open("localhost")
        print("Connected to QLabs")
    except:
        print("Unable to connect to QLabs")
        quit()

    # Delete any previous QCar instances and stop any running spawn models
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    # Spawn a QCar at the given initial pose
    hqcar = QLabsQCar(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        waitForConfirmation=True
    )

    # Create a new camera view and attach it to the QCar
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn()
    # hqcar.possess()
    
    # Spawn traffic elements if requested
    traffic_elements = None
    if spawnTrafficElements:
        traffic_elements = spawn_traffic_elements(qlabs)

    # Start spawn model
    QLabsRealTime().start_real_time_model(rtModel)
    
    print("✅ Setup complete!\n")

    return hqcar, traffic_elements

def terminate():
    QLabsRealTime().terminate_real_time_model("QCar_Workspace")

if __name__ == '__main__':
    # Setup with traffic elements
    hqcar, traffic_elements = setup()
    
    # Example: Access the spawned elements
    if traffic_elements:
        print(f"Spawned {len(traffic_elements['lights'])} traffic lights")
        print(f"Initial light states: {traffic_elements['light_states']}")
    
    # Start cycling traffic lights
    print("\n🚦 Starting traffic light cycling...")
    cycle_traffic_lights(traffic_elements, cycle_interval=8.0)
    
    # Cleanup
    terminate()
    print("✅ Done!\n")