"""
Test script for audio analysis module.
Usage: python test_audio.py <path_to_video_file>
"""

import sys
from pathlib import Path
from app.services.audio import analyze_audio


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_audio.py <path_to_video_file>")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Analyzing video: {video_path}")
    print("-" * 60)

    try:
        result = analyze_audio(video_path)

        print(f"\nTranscript ({len(result['words'])} words):")
        print(f"  {result['transcript']}\n")

        print(f"Speaking Stats:")
        print(f"  Words per minute: {result['words_per_minute']} WPM")
        print(f"  Total speaking time: {result['total_speaking_time']}s\n")

        print(f"Filler Words ({len(result['filler_words'])} detected):")
        if result['filler_words']:
            for fw in result['filler_words']:
                print(f"  [{fw['start']:.1f}s] \"{fw['word']}\"")
        else:
            print("  None detected")

        print("\n" + "-" * 60)
        print("Analysis complete!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
