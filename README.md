# Ophtalomic
GUI application with back-end for AI estimation of human eye refractive errors.
 ____________________________________________________________________________________
The code is not tested on Windows yet. 
The device uses Ximea Camera (model: MQ013RG-ON) a custom made infrared led board (confidential) and medical version of Jetson Tx2NX (forecr). 
To be able to run the code on Jetson nano, Tx2Nx etc do the following:
# Best Parctice:
1- Install Ximea: 
    refer to repository belwo,
        git@github.com:AmirYeganesahab/XimeaPythonInstallation.git
2- Isntall requirements.txt using pip.
3- Connect camera to USB3,
4- Connect LED board to GPIO(UART). check the active UART and if necessary change it in code under [DeviceModules/seonsors.pyx]
  (don't forget to build the packages under device_modules. Setup file is available in the same directory). 
5- Run the main.py or add it to your startup if you want.
