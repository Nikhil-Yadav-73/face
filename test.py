import cv2
import mediapipe as mp
import keyboard
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters
THRESHOLD_Y = 0.5  # Threshold for considering a hand as "up"
COOLDOWN_TIME = 1  # Time in seconds before the same action can be performed again

# Track last executed command and time of last command
last_command = None
last_command_time = 0

# Perform actions based on hand gestures
def execute_command(command):
    global last_command, last_command_time

    current_time = time.time()
    if current_time - last_command_time < COOLDOWN_TIME:
        return  # Prevent re-execution of the same command too quickly

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

    # Update last command and time
    last_command = command
    last_command_time = current_time

# Check if a hand is "up" based on y-coordinate of wrist landmark
def is_hand_up(hand_landmarks):
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    return wrist_y < THRESHOLD_Y

# Check if hands are near the face based on x-coordinate of wrist landmarks
def is_hand_near_face(hand_landmarks):
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    return abs(wrist_x - 0.5) < 0.1  # Assuming face is near the center (x = 0.5)

# Main function
def main():
    global last_command
    cap = cv2.VideoCapture(0)  # Initialize webcam
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error accessing the webcam.")
                break

            # Flip the frame for a selfie view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for hand landmarks
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                left_hand_up = False
                right_hand_up = False
                both_hands_up = False

                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Check for hand positions
                    if is_hand_up(hand_landmarks):
                        if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                            left_hand_up = True  # Left hand detected as up
                        else:
                            right_hand_up = True  # Right hand detected as up
                    else:
                        if abs(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - 0.5) < 0.1:
                            both_hands_up = True  # Both hands are raised

                # Execute corresponding command based on the gesture
                if left_hand_up and right_hand_up:
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

            # Display the video frame
            cv2.imshow("Hand Gesture Control", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
