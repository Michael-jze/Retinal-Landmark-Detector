import cv2
import numpy as np
import math
import argparse

def tmp_plot(img, title, mask=None):
    """
    Display image with optional mask overlay
    img: input image
    title: window title
    mask: optional mask image to overlay on the original image
    """
    img = img.copy()
    if mask is not None:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_image(image_path):
    """
    Load fundus image
    image_path: path to the image file
    returns: loaded image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image, please check the path")
    return img

def calc_valid_mask(img):
    """
    Calculate valid region mask, excluding image edges and background
    img: input image
    returns: valid region mask (255=valid, 0=invalid)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate threshold based on mean and standard deviation
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    threshold = max(10, int(mean_val - std_val))
    
    # Create mask, mark regions below threshold as invalid
    valid_mask = np.ones_like(gray) * 255
    valid_mask[gray < threshold] = 0
    
    print(f"valid_mask info - size: {valid_mask.shape}, mean: {np.mean(valid_mask)}")
    return valid_mask

def detect_disc(img, hsv, valid_mask):
    """
    Detect optic disc region
    img: original image
    hsv: image in HSV color space
    valid_mask: valid region mask
    returns: optic disc region mask
    """
    # Define optic disc color range (HSV space)
    lower_disc = np.array([15, 80, 120])
    upper_disc = np.array([40, 255, 255])
    
    # Color thresholding
    disc_mask = cv2.inRange(hsv, lower_disc, upper_disc)
    
    # Apply valid region mask
    disc_mask = cv2.bitwise_and(disc_mask, valid_mask)
    
    # Gaussian blur to smooth edges
    disc_mask = cv2.GaussianBlur(disc_mask, (5, 5), 0)
    
    # Morphological opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_OPEN, kernel)
    
    # Morphological closing to fill small holes
    kernel = np.ones((7, 7), np.uint8)
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_CLOSE, kernel)
    
    # Binarization
    disc_mask = cv2.threshold(disc_mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    return disc_mask

def locate_disc_center_and_radius(disc_mask, img):
    """
    Locate optic disc center and calculate radius
    disc_mask: optic disc region mask
    img: original image
    returns: optic disc center coordinates and radius
    """
    # Find contours
    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No optic disc detected")
        return None, None
    
    # Select the largest contour as the optic disc
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    
    # Filter out too small regions
    if area < 500:
        print("Warning: Optic disc region too small")
        return None, None
    
    # Calculate minimum enclosing circle to get center and radius
    (center_x, center_y), radius = cv2.minEnclosingCircle(max_contour)
    center = (int(center_x), int(center_y))
    radius = int(radius)
    
    # Visualize optic disc detection result
    result = img.copy()
    cv2.circle(result, center, radius, (0, 255, 0), 2)
    cv2.putText(result, "Optic Disc", (center[0]+radius, center[1]-radius), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    tmp_plot(result, "Detected Optic Disc")
    
    return center, radius

def detect_vessels(img, valid_mask):
    """
    Detect vessel structures in fundus image
    img: original image
    valid_mask: valid region mask
    returns: vessel probability map and binary vessel mask
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply valid region mask
    masked_gray = cv2.bitwise_and(gray, valid_mask)
    
    # Contrast enhancement to highlight vessel structures
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_gray)
    
    # Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Laplacian operator for edge detection, suitable for linear structures (vessels)
    laplacian = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3)
    
    # Thresholding to enhance vessel structures
    _, thresh = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to optimize vessel structure
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    vessels_binary = cv2.erode(dilated, kernel, iterations=1)
    
    # Ensure processing only in valid regions
    vessels_binary = cv2.bitwise_and(vessels_binary, valid_mask)
    
    # Create vessel probability map
    vessels_prob = cv2.normalize(vessels_binary, None, 0, 255, cv2.NORM_MINMAX)
    
    return vessels_prob, vessels_binary

def detect_macula_based_on_disc(img, disc_center, disc_radius, valid_mask):
    """
    Detect macula based on optic disc position
    img: original image
    disc_center: optic disc center coordinates
    disc_radius: optic disc radius
    valid_mask: valid region mask
    returns: macula position coordinates
    """
    if disc_center is None or disc_radius is None:
        print("Error: No optic disc position available, cannot detect macula")
        return None
    
    # Convert to RGB color space (for gray value calculation)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = gray.shape[:2]
    
    # Define search region: circle centered at optic disc with radius 3 times disc radius
    search_radius = min(disc_radius * 3, max(h, w) // 2)
    search_center = disc_center
    
    # Select 360 evenly distributed points on the circle as candidates (1 degree interval)
    num_candidates = 360
    macula_candidates = []
    
    # Detect vessels
    vessels, vessels_binary = detect_vessels(img, valid_mask)
    tmp_plot(img, "Vessels Detection", vessels_binary)
    
    # Create visualization image for candidate points
    candidate_visualization = img.copy()
    
    # Iterate through all candidate points
    for i in range(num_candidates):
        # Calculate point on circle
        angle = 2 * math.pi * i / num_candidates
        x = int(search_center[0] + search_radius * math.cos(angle))
        y = int(search_center[1] + search_radius * math.sin(angle))
        
        # Ensure candidate point is within image bounds
        if 0 <= x < w and 0 <= y < h:
            # Create candidate region mask: circle centered at current point with disc radius
            candidate_mask = np.zeros_like(gray[:, :, 0])
            cv2.circle(candidate_mask, (x, y), disc_radius, 255, -1)
            
            # Apply valid region mask
            masked_candidate = cv2.bitwise_and(candidate_mask, valid_mask)
            
            # Calculate total pixels in candidate region
            total_pixels = np.sum(candidate_mask > 0)
            if total_pixels == 0:
                continue
                
            # Calculate valid region pixels
            valid_pixels = np.sum(masked_candidate > 0)
            valid_area_ratio = valid_pixels / total_pixels
            
            # Continue processing if valid area ratio exceeds threshold
            if valid_area_ratio >= 0.3:
                # Calculate vessel density in candidate region
                vessel_mask = cv2.bitwise_and(vessels_binary, masked_candidate)
                vessel_pixels = np.sum(vessel_mask > 0)
                vessel_density = vessel_pixels / valid_pixels if valid_pixels > 0 else 0
                
                # Calculate mean gray value in candidate region (lower value means darker)
                mean_gray = cv2.mean(gray, mask=masked_candidate)[0]
                
                # Record candidate point information
                macula_candidates.append((x, y, mean_gray, valid_area_ratio, vessel_density))
                cv2.circle(candidate_visualization, (x, y), 3, (255, 0, 255), -1)
    
    # Display all candidate points
    tmp_plot(candidate_visualization, "Macula Candidates")
    
    # Return None if no valid candidates found
    if not macula_candidates:
        print("Warning: No valid macula candidate regions found, try adjusting parameters")
        return None
    
    # Sort by priority:
    # 1. Lowest vessel density (macula region has low vessel density)
    # 2. Lowest gray value (darkest color)
    # 3. Highest valid area ratio
    macula_candidates.sort(key=lambda x: (x[4], x[2], -x[3]))
    best_candidate = macula_candidates[0]
    
    # Visualize final result
    visualization = img.copy()
    cv2.circle(visualization, search_center, search_radius, (255, 0, 0), 2)
    cv2.circle(visualization, (best_candidate[0], best_candidate[1]), disc_radius, (255, 0, 0), 2)
    cv2.putText(visualization, f"Macula (Vessel: {best_candidate[4]:.4f}, Gray: {best_candidate[2]:.1f})", 
               (best_candidate[0]+disc_radius, best_candidate[1]-disc_radius), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    tmp_plot(visualization, "Macula Localization Based on Disc and Vessels")
    
    return (best_candidate[0], best_candidate[1])

def detect_macula_disc(image_path):
    """
    Main function: Detect optic disc and macula in fundus image
    image_path: path to fundus image
    returns: macula position, optic disc position and radius
    """
    # Load image
    img = load_image(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tmp_plot(img, "raw")
    
    # Calculate valid region mask
    valid_mask = calc_valid_mask(img)
    tmp_plot(img, "valid_mask", valid_mask)
    
    # Convert to HSV color space for optic disc detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect optic disc region
    disc_mask = detect_disc(img, hsv, valid_mask)
    tmp_plot(img, "disc_mask", disc_mask)
    
    # Locate optic disc center and calculate radius
    disc_center, disc_radius = locate_disc_center_and_radius(disc_mask, img)
    
    # Detect macula based on optic disc position
    macula_position = detect_macula_based_on_disc(img, disc_center, disc_radius, valid_mask)
    
    # Visualize final result
    result = img.copy()
    if disc_center and disc_radius:
        cv2.circle(result, disc_center, disc_radius, (0, 255, 0), 2)
        cv2.putText(result, "Optic Disc", (disc_center[0]+disc_radius, disc_center[1]-disc_radius), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if macula_position:
        cv2.circle(result, macula_position, 20, (255, 0, 0), 2)
        cv2.putText(result, "Macula", (macula_position[0]+25, macula_position[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    tmp_plot(result, "Final Result")
    
    return macula_position, disc_center, disc_radius

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="images/healthy.png")
    args = parser.parse_args()
    image_path = args.image_path
    
    # Execute optic disc and macula detection
    macula_pos, disc_pos, disc_rad = detect_macula_disc(image_path)
    
    # Output detection results
    if macula_pos:
        print(f"Macula position: x={macula_pos[0]}, y={macula_pos[1]}")
    else:
        print("Failed to detect macula")
    
    if disc_pos:
        print(f"Optic disc position: x={disc_pos[0]}, y={disc_pos[1]}, radius: {disc_rad}")
    else:
        print("Failed to detect optic disc")