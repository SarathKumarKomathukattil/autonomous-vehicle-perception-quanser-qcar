# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
'''
filename.py

High-level summary of the python file with a description of which PDFs
accompany the python example.
'''
import numpy as np
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
import os
#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Main Loop
#=================== Virtual Env Setup ===================
# Connecting to Quanser Interactive Labs
def setup(
        initialPosition=[10, 2, 0.000],
        initialOrientation=[0,0,np.pi/2],
        rtModel = os.path.normpath(
        os.path.join(
        os.path.dirname(__file__),
        '../../spawn_models/QCar_Workspace'
    )
)
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
    hcamera.spawn([8.484, 1.973, 12.209], [-0, 0.748, 0.792])
    hqcar.possess()

    # Start spawn model
    QLabsRealTime().start_real_time_model(rtModel)

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Run
def main():
    setup(initialPosition=[-25.614, 35.574, 0], initialOrientation=[0, 0, 3*np.pi/2])

if __name__ == '__main__':
    main()
#endregion