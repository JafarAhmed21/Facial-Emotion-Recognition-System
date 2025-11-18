import cv2 # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from collections import deque
import time

class RealTimeEmotionDetection:
    def __init__(self, model_path):
        """
        Initialize the emotion detection system
        
        Args:
            model_path (str): Path to your trained .keras model file
        """
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Emotion classes
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # For FPS calculation
        self.frame_times = deque(maxlen=30)
        
        # Create matplotlib figure for real-time plotting
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
    def preprocess_face(self, face_img):
        """Preprocess face image to match training data format"""
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Resize to 48x48 (model's expected input size)
        resized_face = cv2.resize(gray_face, (48, 48))
        # Normalize pixel values
        normalized_face = resized_face / 255.0
        # Add batch and channel dimensions: (1, 48, 48, 1)
        input_face = np.expand_dims(normalized_face, axis=[0, -1])
        return input_face
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        # Preprocess the face
        processed_face = self.preprocess_face(face_img)
        # Make prediction
        predictions = self.model.predict(processed_face, verbose=0)[0]
        # Get the predicted emotion and confidence
        emotion_idx = np.argmax(predictions)
        emotion = self.class_names[emotion_idx]
        confidence = predictions[emotion_idx]
        return emotion, confidence, predictions
    
    def run_detection(self, camera_index=0):
        """
        Run real-time emotion detection
        
        Args:
            camera_index (int): Webcam index (usually 0 for default camera)
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("🎥 Starting real-time emotion detection...")
        print("📹 Press 'q' to quit, 's' to save current frame")
        
        # Create a window
        cv2.namedWindow('Real-time Emotion Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Emotion Detection', 1000, 800)
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                emotions_info = []
                # Process each detected face
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence, all_predictions = self.predict_emotion(face_img)
                    emotions_info.append((emotion, confidence, x, y, w, h))
                    
                    # Choose color based on confidence
                    if confidence > 0.7:
                        color = (0, 255, 0)  # Green - high confidence
                    elif confidence > 0.5:
                        color = (0, 255, 255)  # Yellow - medium confidence
                    else:
                        color = (0, 0, 255)  # Red - low confidence
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Display emotion and confidence
                    label = f"{emotion}: {confidence:.1%}"
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Display face number
                    cv2.putText(frame, f"Face {i+1}", (x, y+h+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Calculate and display FPS
                end_time = time.time()
                frame_time = end_time - start_time
                self.frame_times.append(frame_time)
                fps = 1.0 / frame_time if frame_time > 0 else 0
                avg_fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
                
                # Display FPS information
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Avg FPS: {avg_fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Faces: {len(faces)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display emotions summary
                if emotions_info:
                    emotion_summary = ", ".join([f"{e[0]}({e[1]:.0%})" for e in emotions_info])
                    cv2.putText(frame, f"Emotions: {emotion_summary}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display instructions
                cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the frame
                cv2.imshow('Real-time Emotion Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('s'):  # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"emotion_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Frame saved as: {filename}")
                elif key == ord('p'):  # Pause/play
                    cv2.waitKey(0)  # Wait until any key is pressed
                
        except KeyboardInterrupt:
            print("\n🛑 Detection interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            plt.ioff()
            print("🎬 Emotion detection ended")
    
    def display_emotion_chart(self, predictions):
        """Display emotion probabilities as a bar chart"""
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.class_names, predictions, color='skyblue')
        plt.title('Emotion Probabilities')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, predictions):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    # model file
    MODEL_PATH = "fer_cnn_model_from_scratch.keras" 
    
    try:
        # Create emotion detector instance
        detector = RealTimeEmotionDetection(MODEL_PATH)
        
        # Start real-time detection
        detector.run_detection(camera_index=0)  # Use 0 for default camera
        
    except FileNotFoundError:
        print(f"❌ Model file not found at: {MODEL_PATH}")
        print("Please update the MODEL_PATH variable with the correct path to your .keras model file")
    except Exception as e:
        print(f"❌ Error: {e}")