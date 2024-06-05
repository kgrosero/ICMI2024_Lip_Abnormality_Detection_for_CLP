import numpy as np
import random
import cv2 
from PIL import Image


###########################################################
# Description: Normalize 68 face landmarks based on the interocular distance

def norm_eyes(lks):
    lips_len = np.linalg.norm(lks[54,:] - lks[48,:])
    scaled_lks = lks / lips_len
    translation_vector = np.array([0, 0]) - scaled_lks[48]
    center_lks = scaled_lks + translation_vector
    return center_lks, lips_len, translation_vector

###########################################################
# Description: Combine CL landmarks with control face landmarks

def combine_lks(cl_lks, face_lks):
    cl_rm_lks = np.load(random.choice(cl_lks))  
    # Normalize both control and Cl landmarks   
    sc_lk1, sc1, trans_vec1 = norm_eyes(face_lks)
    sc_lk2, sc2, trans_vec2 = norm_eyes(cl_rm_lks)
    middle_x, middle_y = sc_lk2[49:54,0], sc_lk2[49:54,1]+sc_lk1[51,1]-sc_lk2[51,1]
    middle = np.vstack((middle_x, middle_y)).T
    new_lks = np.concatenate((sc_lk1[:49,:], middle, sc_lk1[54:,:]), axis=0)
    new_lks_sc = np.vstack(((sc1*(new_lks[:,0]-trans_vec1[0])).astype(int),(sc1*(new_lks[:,1]-trans_vec1[1])).astype(int))).T
    return new_lks_sc

###########################################################
# Description: Get points for triangulation

def get_triangulation_indices(points):
    """Get indices triples for every triangle
    """
    #print('points: ', points)
    # Bounding rectangle
    bounding_rect = (*points.min(axis=0), *points.max(axis=0)+1)
    #print(bounding_rect)
    # Triangulate all points
    #try:
    subdiv = cv2.Subdiv2D(bounding_rect)
    subdiv.insert(points.astype(float))

    # Iterate over all triangles
    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():
        # Get index of all points
        yield [(points==point).all(axis=1).nonzero()[0][0] for point in [(x1,y1), (x2,y2), (x3,y3)]]
    #except:
    #    print("error augmenting")
        #print(points)

###########################################################
# Description: Crop to triangle to create small regions

def crop_to_triangle(img, triangle):
    """Crop image to triangle
    """
    # Get bounding rectangle
    bounding_rect = cv2.boundingRect(triangle)
    # Crop image to bounding box
    img_cropped = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                      bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # Move triangle to coordinates in cropped image
    triangle_cropped = [(point[0]-bounding_rect[0], point[1]-bounding_rect[1]) for point in triangle]
    return triangle_cropped, img_cropped
       
###########################################################
# Description: Apply affine transformations for CL transformation 

def transform(src_img, src_points, dst_img, dst_points):
    """Transforms source image to target image, overwriting the target image.
    """
    for indices in get_triangulation_indices(src_points):
        #print('indices: ', indices)
        # Get triangles from indices
        src_triangle = src_points[indices]
        dst_triangle = dst_points[indices]

        # Crop to triangle, to make calculations more efficient
        src_triangle_cropped, src_img_cropped = crop_to_triangle(src_img, src_triangle)
        dst_triangle_cropped, dst_img_cropped = crop_to_triangle(dst_img, dst_triangle)

        # Calculate transfrom to warp from old image to new
        transform = cv2.getAffineTransform(np.float32(src_triangle_cropped), np.float32(dst_triangle_cropped))        
        
        # Warp image
        dst_img_warped = cv2.warpAffine(src_img_cropped, transform, (dst_img_cropped.shape[1], dst_img_cropped.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

        # Create mask for the triangle we want to transform
        mask = np.zeros(dst_img_cropped.shape, dtype = np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_triangle_cropped), (1.0, 1.0, 1.0), 16, 0);

        # Delete all existing pixels at given mask
        dst_img_cropped*=1-mask
        # Add new pixels to masked area
        dst_img_cropped+=dst_img_warped*mask

###########################################################
# Description: Main function for CLP transformation whole process
# returns the transformed image and landmarks
    
def func_transform_cl(cl_lks, image, landmark):
    warp_lm = landmark.copy()
    dest_img = image.copy()
    # transformation options
    warp_lm = combine_lks(cl_lks, warp_lm)
    # Transform image  
    transform(image, landmark, dest_img, warp_lm)
    out_img = Image.fromarray(dest_img, "RGB")
    return out_img, warp_lm  

def face_lks(name, predictor, face_detector):
    frame = cv2.imread(name)
    faces = face_detector(frame)
    for face in faces:
        landmarks = predictor(frame, face)                
        points = []
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y  
            points.append((x, y))
        points = np.array(points, np.int32)
        return points
