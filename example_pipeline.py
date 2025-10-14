"""
Example: Complete Surveillance Pipeline
Demonstrates how to use all nodes together
"""

import os
from pathlib import Path

# Import all nodes
from nodes import (
    FrameExtractorNode,
    ObjectDetectionNode,
    TrackingNode,
    AlertNode,
    VLMNode
)


def run_surveillance_pipeline(video_path: str, output_dir: str = "surveillance_storage", 
                               stride: int = 2, use_vlm: bool = False):
    """
    Run complete surveillance pipeline on a video
    
    Args:
        video_path: Path to input video file
        output_dir: Output directory for surveillance storage
        stride: Frame extraction stride (1=all frames, 2=every other frame, etc.)
        use_vlm: Whether to run VLM analysis (requires GEMINI_API_KEY)
    
    Returns:
        session_id: Generated session identifier
    """
    
    print("=" * 60)
    print("🔒 SURVEILLANCE PIPELINE")
    print("=" * 60)
    
    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # ===== STEP 1: Extract Frames =====
    print("\n📹 Step 1: Extracting frames...")
    frame_extractor = FrameExtractorNode()
    frames, frames_meta = frame_extractor.process(video_path, output_dir, stride=stride)
    
    session_id = frames_meta[0]['session_id']
    print(f"   ✓ Extracted {len(frames)} frames")
    print(f"   ✓ Session ID: {session_id}")
    
    # ===== STEP 2: Detect Objects =====
    print("\n🔍 Step 2: Detecting objects...")
    detector = ObjectDetectionNode({
        'conf_threshold': 0.5
    })
    classes, bboxes, confs = detector.process(frames, frames_meta)
    
    total_detections = sum(len(c) for c in classes)
    print(f"   ✓ Found {total_detections} detections")
    
    # Save annotated frames
    detector.save_annotated_frames(frames, frames_meta, classes, bboxes, confs, output_dir)
    print(f"   ✓ Saved annotated frames")
    
    # ===== STEP 3: Track Objects =====
    print("\n🎯 Step 3: Tracking objects...")
    tracker = TrackingNode({
        'iou_threshold': 0.3,
        'max_age': 30
    })
    tracks, = tracker.process(classes, bboxes, confs, frames_meta)
    
    unique_tracks = set()
    for frame_tracks in tracks:
        for track in frame_tracks:
            unique_tracks.add(track['track_id'])
    
    print(f"   ✓ Tracked {len(unique_tracks)} unique objects")
    
    # ===== STEP 4: Generate Alerts =====
    print("\n🚨 Step 4: Generating alerts...")
    alert_gen = AlertNode({
        'rules': {
            'loitering': {
                'enabled': True,
                'duration_threshold': 30.0,
                'severity': 'Medium'
            },
            'suspicious_object': {
                'enabled': True,
                'watchlist': ['knife', 'gun', 'rifle', 'pistol', 'weapon'],
                'conf_threshold': 0.5,
                'severity': 'Critical'
            },
            'crowding': {
                'enabled': True,
                'person_threshold': 10,
                'severity': 'Medium'
            }
        }
    })
    alerts, = alert_gen.process(classes, bboxes, confs, tracks, frames_meta)
    
    print(f"   ✓ Generated {len(alerts)} alerts")
    
    # Print alert summary
    if alerts:
        severity_counts = {}
        for alert in alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        print("\n   Alert Summary:")
        for severity, count in severity_counts.items():
            print(f"     - {severity}: {count}")
    
    # ===== STEP 5: VLM Analysis (Optional) =====
    report = None
    if use_vlm:
        print("\n🤖 Step 5: Running VLM analysis...")
        vlm = VLMNode({'session_id': session_id})
        
        try:
            report, = vlm.process(video_path)
            
            if report and 'summary_for_user' in report:
                summary = report['summary_for_user']
                print(f"   ✓ Risk Level: {summary.get('overall_risk_level', 'Unknown')}")
                print(f"   ✓ Justification: {summary.get('justification', 'N/A')[:100]}...")
        except Exception as e:
            print(f"   ⚠ VLM analysis failed: {e}")
            print(f"   💡 Tip: Set GEMINI_API_KEY environment variable")
    else:
        print("\n⏭  Step 5: Skipping VLM analysis (use --vlm to enable)")
    
    # ===== STEP 6: Save All Metadata =====
    print("\n💾 Step 6: Saving metadata...")
    session_path = f"{output_dir}/session_{session_id}"
    
    detector.save_metadata(session_path)
    tracker.save_metadata(session_path)
    alert_gen.save_metadata(session_path)
    
    if report:
        vlm.save_metadata(session_path)
    
    print(f"   ✓ Metadata saved to: {session_path}/metadata/")
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"📊 Session ID: {session_id}")
    print(f"🎞  Frames: {len(frames)}")
    print(f"🔍 Detections: {total_detections}")
    print(f"🎯 Unique Tracks: {len(unique_tracks)}")
    print(f"🚨 Alerts: {len(alerts)}")
    print(f"📁 Output: {session_path}/")
    print("\n💡 Next steps:")
    print(f"   1. Start dashboard: python dashboard/backend.py")
    print(f"   2. Open: dashboard/frontend.html in your browser")
    print(f"   3. Select session: {session_id}")
    print("=" * 60)
    
    return session_id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run surveillance pipeline on video')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', default='surveillance_storage', help='Output directory')
    parser.add_argument('--stride', type=int, default=2, help='Frame extraction stride')
    parser.add_argument('--vlm', action='store_true', help='Enable VLM analysis')
    
    args = parser.parse_args()
    
    try:
        session_id = run_surveillance_pipeline(
            args.video,
            output_dir=args.output,
            stride=args.stride,
            use_vlm=args.vlm
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
