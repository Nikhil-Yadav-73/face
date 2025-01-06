import cv2
import mediapipe as mp
import keyboard
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

THRESHOLD_Y = 0.5
COOLDOWN_TIME = 1

last_command = None
last_command_time = 0

def execute_command(command):
    global last_command, last_command_time

    current_time = time.time()
    if current_time - last_command_time < COOLDOWN_TIME:
        return

    if command == "volume down":
        print("Decreasing volume...")
        keyboard.press_and_release("volume down")
    elif command == "volume up":
        print("Increasing volume...")
        keyboard.press_and_release("volume up")
    elif command == "play/pause":
        print("Toggling play/pause...")
        keyboard.press_and_release("play/pause media")
    else:
        print("No action for this gesture.")

    last_command = command
    last_command_time = current_time

def is_hand_up(hand_landmarks):
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    return wrist_y < THRESHOLD_Y

def is_hand_near_face(hand_landmarks):
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    return abs(wrist_x - 0.5) < 0.1


def main():
    global last_command
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error accessing the webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                left_hand_up = False
                right_hand_up = False
                both_hands_up = False

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if is_hand_up(hand_landmarks):
                        if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                            left_hand_up = True
                        else:
                            right_hand_up = True
                    else:
                        if abs(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - 0.5) < 0.1:
                            both_hands_up = True

                if (left_hand_up and right_hand_up) or both_hands_up:
                    print("Both hands up detected!")
                    execute_command("play/pause")
                elif left_hand_up:
                    print("Left hand up detected!")
                    execute_command("volume down")
                elif right_hand_up:
                    print("Right hand up detected!")
                    execute_command("volume up")
                else:
                    print("No significant hand gesture detected.")

            cv2.imshow("Hand Gesture Control", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()