import cv2
import sys
import time
import numpy as np
from pathlib import Path

class FastImageTracker:
    """High-performance image-based object tracker with comprehensive algorithm options"""
    
    # Available tracker types categorized by approach
    TRACKER_TYPES = {
        # Correlation-based trackers
        'MOSSE': {'create': cv2.TrackerMOSSE_create, 'category': 'Correlation', 'speed': 'Very Fast', 'desc': 'Minimum Output Sum of Squared Error'},
        'KCF': {'create': cv2.TrackerKCF_create, 'category': 'Correlation', 'speed': 'Fast', 'desc': 'Kernelized Correlation Filter'},
        'CSRT': {'create': cv2.TrackerCSRT_create, 'category': 'Correlation', 'speed': 'Moderate', 'desc': 'Discriminative Correlation Filter with Channel and Spatial Reliability'},
        
        # Learning-based trackers
        'MIL': {'create': cv2.TrackerMIL_create, 'category': 'Learning', 'speed': 'Moderate', 'desc': 'Multiple Instance Learning'},
        'BOOSTING': {'create': cv2.legacy.TrackerBoosting_create, 'category': 'Learning', 'speed': 'Slow', 'desc': 'AdaBoost-based tracker'},
        'TLD': {'create': cv2.legacy.TrackerTLD_create, 'category': 'Learning', 'speed': 'Slow', 'desc': 'Tracking, Learning and Detection'},
        'MEDIANFLOW': {'create': cv2.legacy.TrackerMedianFlow_create, 'category': 'Optical Flow', 'speed': 'Fast', 'desc': 'Median Flow tracker'},
        
        # Feature-based tracker
        'GOTURN': {'create': lambda: cv2.TrackerGOTURN_create() if hasattr(cv2, 'TrackerGOTURN_create') else None, 
                   'category': 'Deep Learning', 'speed': 'Fast', 'desc': 'Generic Object Tracking Using Regression Networks (needs model)'},
    }
    
    # Multi-object tracking strategies
    MULTI_TRACKING_MODES = {
        'INDEPENDENT': 'Track each object independently',
        'SHARED_FEATURES': 'Share feature computation across objects',
    }
    
    def __init__(self, tracker_type='KCF', source=0, multi_object=False):
        """
        Initialize tracker
        Args:
            tracker_type: Algorithm type
            source: 0 for webcam, or path to video file
            multi_object: Enable multi-object tracking
        """
        self.tracker_type = tracker_type
        self.source = source
        self.multi_object = multi_object
        self.trackers = []  # List of (tracker, bbox, id, color) tuples
        self.next_id = 1
        self.tracking = False
        self.fps_list = []
        self.colors = self.generate_colors(20)
        
    def generate_colors(self, n):
        """Generate distinct colors for multiple objects"""
        np.random.seed(42)
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def create_tracker(self):
        """Create a new tracker instance"""
        try:
            tracker_info = self.TRACKER_TYPES[self.tracker_type]
            tracker = tracker_info['create']()
            if tracker is None:
                print(f"Warning: {self.tracker_type} not available, falling back to KCF")
                tracker = cv2.TrackerKCF_create()
                self.tracker_type = 'KCF'
            return tracker
        except Exception as e:
            print(f"Error creating tracker {self.tracker_type}: {e}")
            print("Falling back to KCF")
            self.tracker_type = 'KCF'
            return cv2.TrackerKCF_create()
    
    def select_roi(self, frame, roi_id=None):
        """Let user select region of interest"""
        title = f"Select Object #{roi_id}" if roi_id else "Select Object"
        print(f"\n=== ROI Selection {f'(Object #{roi_id})' if roi_id else ''} ===")
        print("1. Draw a box around the object to track")
        print("2. Press ENTER to confirm")
        print("3. Press C to cancel")
        
        bbox = cv2.selectROI(title, frame, False, False)
        cv2.destroyWindow(title)
        
        if bbox[2] > 0 and bbox[3] > 0:
            return bbox
        return None
    
    def draw_info(self, frame, fps, num_tracked=0):
        """Draw tracking info on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for info panel
        overlay = frame.copy()
        panel_height = 140 if self.multi_object else 110
        cv2.rectangle(overlay, (10, 10), (350, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Get tracker info
        tracker_info = self.TRACKER_TYPES.get(self.tracker_type, {})
        category = tracker_info.get('category', 'Unknown')
        speed = tracker_info.get('speed', 'Unknown')
        
        # Draw text info
        y_pos = 35
        cv2.putText(frame, f"Algorithm: {self.tracker_type} ({category})", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        y_pos += 25
        cv2.putText(frame, f"Speed Class: {speed}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        y_pos += 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        if self.multi_object:
            y_pos += 25
            cv2.putText(frame, f"Objects: {num_tracked}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        # Instructions
        if self.multi_object:
            instructions = "R: Reset | A: Add | D: Delete | Q: Quit"
        else:
            instructions = "R: Reset | Q: Quit | Space: Pause"
        cv2.putText(frame, instructions, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_bbox(self, frame, bbox, obj_id, color, success):
        """Draw bounding box on frame"""
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            
            # Draw main box
            box_color = color if success else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            
            # Draw corner markers
            corner_len = 20
            thickness = 4
            # Top-left
            cv2.line(frame, (x, y), (x + corner_len, y), box_color, thickness)
            cv2.line(frame, (x, y), (x, y + corner_len), box_color, thickness)
            # Top-right
            cv2.line(frame, (x + w, y), (x + w - corner_len, y), box_color, thickness)
            cv2.line(frame, (x + w, y), (x + w, y + corner_len), box_color, thickness)
            # Bottom-left
            cv2.line(frame, (x, y + h), (x + corner_len, y + h), box_color, thickness)
            cv2.line(frame, (x, y + h), (x, y + h - corner_len), box_color, thickness)
            # Bottom-right
            cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), box_color, thickness)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), box_color, thickness)
            
            # Draw center point
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), 5, box_color, -1)
            
            # Show ID and status
            status = "OK" if success else "LOST"
            label = f"ID:{obj_id} ({cx},{cy}) [{status}]"
            
            # Background for label
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 10, y), box_color, -1)
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def add_object(self, frame, bbox):
        """Add a new object to track"""
        tracker = self.create_tracker()
        tracker.init(frame, bbox)
        color = self.colors[self.next_id % len(self.colors)]
        self.trackers.append({
            'tracker': tracker,
            'bbox': bbox,
            'id': self.next_id,
            'color': color,
            'success': True
        })
        print(f"Added object #{self.next_id}")
        self.next_id += 1
    
    def remove_failed_objects(self):
        """Remove objects that failed to track"""
        original_count = len(self.trackers)
        self.trackers = [t for t in self.trackers if t['success']]
        removed = original_count - len(self.trackers)
        if removed > 0:
            print(f"Removed {removed} lost object(s)")
    
    def run_single_object(self):
        """Single object tracking mode"""
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
            return
        
        source_type = "Webcam" if self.source == 0 else Path(str(self.source)).name
        print(f"\n=== Single Object Tracking ===")
        print(f"Source: {source_type}")
        print(f"Algorithm: {self.tracker_type}")
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            cap.release()
            return
        
        # Select initial ROI
        bbox = self.select_roi(frame)
        if bbox is None:
            print("No object selected. Exiting...")
            cap.release()
            return
        
        # Initialize tracker
        tracker = self.create_tracker()
        tracker.init(frame, bbox)
        self.tracking = True
        
        paused = False
        success = True
        
        # Main loop
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or read error")
                    break
                
                start_time = time.time()
                
                # Update tracker
                if self.tracking:
                    success, bbox = tracker.update(frame)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_list.append(fps)
                if len(self.fps_list) > 30:
                    self.fps_list.pop(0)
                avg_fps = sum(self.fps_list) / len(self.fps_list)
                
                # Draw bbox and info
                frame = self.draw_bbox(frame, bbox, 1, (0, 255, 0), success)
                frame = self.draw_info(frame, avg_fps)
            
            cv2.imshow("Image-Based Tracker", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                bbox = self.select_roi(frame)
                if bbox is not None:
                    tracker = self.create_tracker()
                    tracker.init(frame, bbox)
                    self.tracking = True
                    self.fps_list.clear()
                    paused = False
            elif key == ord(' '):
                paused = not paused
        
        if self.fps_list:
            print(f"\nAverage FPS: {sum(self.fps_list) / len(self.fps_list):.2f}")
        cap.release()
        cv2.destroyAllWindows()
    
    def run_multi_object(self):
        """Multi-object tracking mode"""
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
            return
        
        source_type = "Webcam" if self.source == 0 else Path(str(self.source)).name
        print(f"\n=== Multi-Object Tracking ===")
        print(f"Source: {source_type}")
        print(f"Algorithm: {self.tracker_type}")
        print("\nControls:")
        print("  A - Add new object")
        print("  D - Remove lost objects")
        print("  R - Reset all")
        print("  Q - Quit")
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            cap.release()
            return
        
        # Select initial objects
        print("\nSelect initial objects (press ESC when done):")
        while True:
            bbox = self.select_roi(frame, self.next_id)
            if bbox is None:
                break
            self.add_object(frame, bbox)
        
        if not self.trackers:
            print("No objects selected. Exiting...")
            cap.release()
            return
        
        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or read error")
                break
            
            start_time = time.time()
            
            # Update all trackers
            for tracker_obj in self.trackers:
                success, bbox = tracker_obj['tracker'].update(frame)
                tracker_obj['bbox'] = bbox
                tracker_obj['success'] = success
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_list.append(fps)
            if len(self.fps_list) > 30:
                self.fps_list.pop(0)
            avg_fps = sum(self.fps_list) / len(self.fps_list)
            
            # Draw all bboxes
            for tracker_obj in self.trackers:
                frame = self.draw_bbox(frame, tracker_obj['bbox'], tracker_obj['id'], 
                                      tracker_obj['color'], tracker_obj['success'])
            
            frame = self.draw_info(frame, avg_fps, len(self.trackers))
            cv2.imshow("Multi-Object Tracker", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a'):
                bbox = self.select_roi(frame, self.next_id)
                if bbox is not None:
                    self.add_object(frame, bbox)
            elif key == ord('d'):
                self.remove_failed_objects()
            elif key == ord('r'):
                self.trackers.clear()
                self.next_id = 1
                self.fps_list.clear()
                print("\nReset complete. Select new objects:")
                while True:
                    bbox = self.select_roi(frame, self.next_id)
                    if bbox is None:
                        break
                    self.add_object(frame, bbox)
        
        if self.fps_list:
            print(f"\nAverage FPS: {sum(self.fps_list) / len(self.fps_list):.2f}")
        cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Main entry point"""
        if self.multi_object:
            self.run_multi_object()
        else:
            self.run_single_object()


def main():
    """Entry point with argument parsing"""
    print("=" * 60)
    print("   COMPREHENSIVE IMAGE-BASED OBJECT TRACKING SYSTEM")
    print("=" * 60)
    
    # Display all available trackers
    print("\n📊 Available Tracking Algorithms:\n")
    
    categories = {}
    for i, (name, info) in enumerate(FastImageTracker.TRACKER_TYPES.items(), 1):
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((i, name, info))
    
    tracker_map = {}
    idx = 1
    for cat, trackers in categories.items():
        print(f"\n{cat}-Based Trackers:")
        for i, name, info in trackers:
            print(f"  {idx}. {name:12} - {info['speed']:12} | {info['desc']}")
            tracker_map[str(idx)] = name
            idx += 1
    
    print("\nRecommended:")
    print("  • Fast & Accurate: KCF or CSRT")
    print("  • Maximum Speed: MOSSE or MEDIANFLOW")
    print("  • Best Accuracy: CSRT")
    
    # Select tracker
    choice = input(f"\nSelect tracker (1-{len(tracker_map)}) [default: 2 (KCF)]: ").strip()
    tracker_type = tracker_map.get(choice, 'KCF')
    
    # Select tracking mode
    print("\n📹 Tracking Mode:")
    print("1. Single Object Tracking")
    print("2. Multi-Object Tracking")
    mode_choice = input("Select mode (1-2) [default: 1]: ").strip()
    multi_object = (mode_choice == '2')
    
    # Select source
    print("\n🎥 Video Source:")
    print("1. Webcam")
    print("2. Video File")
    source_choice = input("Select source (1-2) [default: 1]: ").strip()
    
    if source_choice == '2':
        video_path = input("Enter video file path: ").strip()
        if not Path(video_path).exists():
            print(f"Error: File not found: {video_path}")
            return
        source = video_path
    else:
        source = 0
    
    # Run tracker
    print("\n🚀 Initializing tracker...")
    tracker = FastImageTracker(tracker_type=tracker_type, source=source, multi_object=multi_object)
    tracker.run()


if __name__ == "__main__":
    main()