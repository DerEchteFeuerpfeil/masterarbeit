import os
import tempfile
import svgpathtools
from tqdm import tqdm
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import deepsvg

def svg_to_frames(svg_file_path):
    # Parse the SVG file and extract paths
    paths, attr, svg_attr = svgpathtools.svg2paths2(svg_file_path)

    # Temporary directory to store frames
    frames_dir = tempfile.mkdtemp()

    # Determine the bounding box of all paths to set consistent plot limits
    min_x, min_y, max_x, max_y = [float(x) for x in svg_attr["viewBox"].split(" ")]

    # Initialize plot with a fixed size and limits
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Set consistent axis limits based on the bounding box of the SVG
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Frame counter
    frame_number = 0
        
    # Clear the plot
    ax.cla()
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Keep the limits consistent
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    ax.invert_yaxis()
    
    # Draw each path as it is added
    all_segments = [segment for path in paths for segment in path._segments]

    # Save the current frame as an image - start with blanc frame
    frame_path = os.path.join(frames_dir, f"frame_{frame_number:04d}.png")
    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0, dpi=300)
    frame_number += 1

    for segment in tqdm(all_segments):
        if isinstance(segment, svgpathtools.Line):
            ax.plot([segment.start.real, segment.end.real], 
                    [segment.start.imag, segment.end.imag], 
                    color='black', lw=2)
        elif isinstance(segment, svgpathtools.CubicBezier):
            # Plot CubicBezier curve by discretizing it
            bezier_points = [segment.start] + [segment.point(t) for t in [0.25, 0.5, 0.75]] + [segment.end]
            ax.plot([p.real for p in bezier_points], [p.imag for p in bezier_points], 
                    color='black', lw=2)
        elif isinstance(segment, svgpathtools.QuadraticBezier):
            # Plot QuadraticBezier curve by discretizing it
            bezier_points = [segment.start] + [segment.point(t) for t in [0.25, 0.5, 0.75]] + [segment.end]
            ax.plot([p.real for p in bezier_points], [p.imag for p in bezier_points], 
                    color='black', lw=2)
        elif isinstance(segment, svgpathtools.Arc):
            # Approximate Arc with a few line segments
            arc_points = [segment.point(t) for t in [0, 0.25, 0.5, 0.75, 1]]
            ax.plot([p.real for p in arc_points], [p.imag for p in arc_points], 
                    color='black', lw=2)
        else:
            print(f"Unknown segment type: {type(segment)}")

        # Save the current frame as an image
        frame_path = os.path.join(frames_dir, f"frame_{frame_number:04d}.png")
        plt.savefig(frame_path, bbox_inches='tight', pad_inches=0, dpi=300)
        frame_number += 1

    plt.close(fig)

    return frames_dir, frame_number

def create_video_gif(frames_dir, total_frames, output_mp4, output_gif, fps=10, effect=None):
    # Get the list of frames
    frames = [os.path.join(frames_dir, f"frame_{i:04d}.png") for i in range(total_frames)]

    # Create a video (mp4)
    if effect == "reverse":
        clip = ImageSequenceClip(frames + frames[::-1], fps=fps)
    else:
        clip = ImageSequenceClip(frames, fps=fps)

    clip.write_videofile(output_mp4, codec='libx264')

    # Create a gif
    clip.write_gif(output_gif, fps=fps)

def main(svg_file_path, output_mp4, output_gif, fps=10, effect=None):
    # Convert SVG to frames
    frames_dir, total_frames = svg_to_frames(svg_file_path)

    # Create video and GIF
    create_video_gif(frames_dir, total_frames, output_mp4, output_gif, fps, effect=effect)

    # Cleanup
    for frame_file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame_file))
    os.rmdir(frames_dir)

# Example usage:
svg_file = "/scratch2/moritz_logs/thesis/Stage2_figr8/nseg=1_ncode=2_lseg=8_no_shuffle/test/vq_context_0/svgs/sample_3.svg"
output_mp4 = "test.mp4"
output_gif = "test.gif"

main(svg_file, output_mp4, output_gif, effect="reverse")
