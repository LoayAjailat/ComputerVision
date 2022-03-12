import cv2
import cvzone
import mediapipe

class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

class Keyboard:
    def __init__(self) -> None:
        self.buttonList = []
        self.createLayout()

    def createLayout(self):
        keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", " "]]

        for i in range(len(keys)):
            for j, key in enumerate(keys[i]):
                self.buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

    def draw(self, img):
        for button in self.buttonList:
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

def runVirtualKeyboard():
    o_Keyboard = Keyboard()
    buttonList = o_Keyboard.buttonList
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
                                        print(button.text)
                                        textString += button.text
                            else:
                                if abs(middle[0] - index[0]) > 40:
                                    pressed = False
                                    break
            # Draw the keyboard
            img = o_Keyboard.draw(img)

            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF

            # Below states that if the |q| is press on the keyboard it will stop the system
            if key == ord("q"):
                print(textString)
                break

if __name__ == "__main__":
    runVirtualKeyboard()