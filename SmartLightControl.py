import cv2
import cvzone
import mediapipe

import asyncio
import os
from random import randint

from meross_iot.http_api import MerossHttpClient
from meross_iot.manager import MerossManager
from meross_iot.model.enums import OnlineStatus

# TODO: Explain the .env file
EMAIL = os.environ.get('MEROSS_EMAIL')
PASSWORD = os.environ.get('MEROSS_PASSWORD')

async def connect():
    global plugs, manager, http_api_client
    # Setup the HTTP client API from user-password
    http_api_client = await MerossHttpClient.async_from_user_password(email=EMAIL, password=PASSWORD)

    # Setup and start the device manager
    manager = MerossManager(http_client=http_api_client)
    await manager.async_init()

    # Retrieve the MSL120 devices that are registered on this account
    await manager.async_device_discovery()
    plugs = manager.find_devices(device_type="msl430", online_status=OnlineStatus.ONLINE)

async def main(colour):
    if len(plugs) < 1:
        print("No online msl430 smart bulbs found...")
    else:
        # Let's play with RGB colors. Note that not all light devices will support
        # rgb capabilities. For this reason, we first need to check for rgb before issuing
        # color commands.
        dev = plugs[0]

        # Update device status: this is needed only the very first time we play with this device (or if the
        #  connection goes down)
        await dev.async_update()
        if not dev.get_supports_rgb():
            print("Unfortunately, this device does not support RGB...")
        else:
            # Check the current RGB color
            current_color = dev.get_rgb_color()
            print(f"Currently, device {dev.name} is set to color (RGB) = {current_color}")
            # Randomly chose a new color
            # rgb = randint(0, 255), randint(0, 255), randint(0, 255)
            # print(f"Chosen random color (R,G,B): {rgb}")
            rgb = colour
            await dev.async_set_light_color(rgb=rgb)
            print("Color changed!")

async def end():
    global plugs, manager, http_api_client
    # Close the manager and logout from http_api
    print("Closing manager and logging out.")
    manager.close()
    await http_api_client.async_logout()

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", " "]]
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        # Draw a green L on the corners
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        # Draw the rectangles for each key
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        # Add text to each rectangle
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

if __name__ == '__main__':
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(connect())
    # Use MediaPipe to draw the hand framework on top of hands it identifies in real-time
    drawingModule = mediapipe.solutions.drawing_utils
    handsModule = mediapipe.solutions.hands

    WEBCAM = 0
    cap = cv2.VideoCapture(WEBCAM, cv2.CAP_DSHOW)  # Using cv2.CAP_DSHOW removes any delays in starting the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Sets the width of the captured frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Sets the height of the captured frame
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    textString = ""
    pressed = False  # Flag to set if a key has been pressed

    # Add confidence values and extra settings to MediaPipe hand tracking. As we are using a live video stream this is not a static
    # image mode, confidence values in regards to overall detection and tracking and we will only let two hands be tracked at the same time
    # More hands can be tracked at the same time if desired but will slow down the system
    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        while True:
            success, img = cap.read()
            FLAG_Y = 1
            img = cv2.flip(img, FLAG_Y)  # Flip image horizontally so that the mirrored image moves to the 'same side'

            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Incase the system sees multiple hands this if statment deals with that and produces another hand overlay
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)

                    # Get normalized coordinates
                    indexNormalized = handLandmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_TIP]
                    middleNormalized = handLandmarks.landmark[handsModule.HandLandmark.MIDDLE_FINGER_TIP]
                    # Get pixel coordinates
                    index = drawingModule._normalized_to_pixel_coordinates(indexNormalized.x, indexNormalized.y, 1280, 720)
                    middle = drawingModule._normalized_to_pixel_coordinates(middleNormalized.x, middleNormalized.y, 1280, 720)
                    if index != None and middle != None:
                        for button in buttonList:
                            x, y = button.pos
                            w, h = button.size
                            if not pressed:
                                # Check if index is within the button's rect
                                if x <= index[0] <= (x + w) and y <= index[1] <= (y + h):
                                    # Check the distance between the index and middle finger
                                    if abs(middle[0] - index[0]) <= 30 and abs(middle[1] - index[1]) <= 30:
                                        pressed = True
                                        if button.text == "R":
                                            colour = (255,0,0)
                                            loop.run_until_complete(main(colour))
                                        elif button.text == "G":
                                            colour = (0,255,0)
                                            loop.run_until_complete(main(colour))
                                        elif button.text == "B":
                                            colour = (0,0,255)
                                            loop.run_until_complete(main(colour))
                                        print(button.text)
                                        textString += button.text
                            else:
                                if abs(middle[0] - index[0]) > 40:
                                    pressed = False
                                    break
            # Draw the keyboard
            img = drawAll(img, buttonList)

            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF

            # Below states that if the |q| is press on the keyboard it will stop the system
            if key == ord("q"):
                print(textString)
                loop.run_until_complete(end())
                loop.close()
                break
