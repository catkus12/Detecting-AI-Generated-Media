import cv2
import numpy as np

def detect_artifacts(image_path):
    """Detects artifacts/blurriness using adaptive Laplacian variance."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    
    # Dynamic threshold based on median sharpness of natural images
    threshold = max(100, laplacian * 0.15)  # Adjust threshold dynamically
    return laplacian < threshold, laplacian  

def detect_fine_details(image_path):
    """Detects lack of fine details using edge density, normalized."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150)  # Adjusted Canny parameters
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    # Normalize edge density for better AI detection
    normalized_edge_density = edge_density / 255  
    return normalized_edge_density < 0.02, normalized_edge_density

def detect_lighting_inconsistencies(image_path):
    """Analyzes unnatural brightness distributions using histogram variance."""
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    _, _, v = cv2.split(image_hsv)
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

    hist_variance = np.var(hist_v)
    
    # AI images may have extremely low OR high variance, check for outliers
    if hist_variance < 50000 or hist_variance > 2000000:
        return True, hist_variance  
    return False, hist_variance

def analyze_image(image_path):
    """Runs all three detection functions and determines AI vs. real."""
    artifacts, laplacian_value = detect_artifacts(image_path)
    fine_details, edge_density = detect_fine_details(image_path)
    lighting, hist_variance = detect_lighting_inconsistencies(image_path)
    
    # Debugging output to analyze failures
    print(f"Debug - Laplacian Variance: {laplacian_value:.2f}, Edge Density: {edge_density:.6f}, Histogram Variance: {hist_variance:.2f}")
    
    print("Analysis Results:")
    print(f"Artifacts Detected: {artifacts} (Laplacian Variance: {laplacian_value:.2f})")
    print(f"Lack of Fine Details: {fine_details} (Normalized Edge Density: {edge_density:.6f})")
    print(f"Lighting Inconsistencies: {lighting} (Histogram Variance: {hist_variance:.2f})")
    
    ai_score = sum([artifacts, fine_details, lighting])
    
    if ai_score >= 2:
        print("Final Verdict: The image is likely AI_generated.")
    else:
        print("Final Verdict: The image is likely Cat-generated (human-generated).")

if __name__ == "__main__":
    print("Deena")
    image_path = "Deena.jpg"  
    analyze_image(image_path)

    print(f"Gordon")
    image_path = "Gordon.jpg"  
    analyze_image(image_path)

    print(f"Kiki")
    image_path = "Kiki.jpg"  
    analyze_image(image_path)

    print(f"GPTCat")
    image_path = "GPTCat.jpg"  
    analyze_image(image_path)

    print(f"DeepAICat")
    image_path = "DeepAICat.jpg" 
    analyze_image(image_path)

    print(f"LeonardoCat")
    image_path = "LeonardoCat.jpeg"  
    analyze_image(image_path)
