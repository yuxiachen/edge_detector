import matplotlib.image as mpimg
import argparse

from edge_detector import *

# Funtion to read different type of image 
# such as RGB image, gray image, binary image
def read_img(filename):
    img = mpimg.imread(filename)
    # convert to gray image
    gimg = img[:, :, 0]
    if np.array_equal(gimg, gimg >= 1):
    	# binary image
    	return gimg

    if np.array_equal(gimg, np.uint8(gimg)):
    	# gray image
    	return gimg

    assert gimg.min() >= 0 and gimg.max() <= 1, gimg
    gimg = gimg * 255
    
    return np.array(gimg, dtype='uint8')

#function to save the image
def save_img(img, filename):
    mpimg.imsave(filename, img, cmap='gray')

# The dictionary of operators applied on the image
FUNC_MAP = {
	'forward_operator': forward_operator,
	'backward_operator': backward_operator,
	'central_operator': central_operator,
	'sobel_operator': sobel_operator,
	'prewitt_operator': prewitt_operator,
	'canny_operator': canny_operator,
	'laplacian_operator': laplacian_operator,
	'log': LoG,
	'dog': DoG,
	'binarize_operation': binarize_operation,
	'dilation_operation': dilation_operation,
	'erosion_operation': erosion_operation,
	'open_operation': open_operation,
	'close_operation': close_operation
}

# Main function
def main(args):
	image_file = args.image_file
	output_file = args.output_file
	method = args.method.lower()
	
	func = FUNC_MAP[method]

	img = read_img(image_file)

	print('processing image using {} method'.format(method))
	if method == 'log':
		sigma = args.sigma
		assert sigma is not None, 'need sigma for LoG'
		ret = func(img, sigma)
	elif method == 'dog':
		sigma1 = args.sigma
		sigma2 = args.sigma2
		ret = func(img, sigma1, sigma2)
	elif method == 'binarize_operation':
		threshold = args.threshold
		ret = func(img, threshold)
	else:
		ret = func(img)

	print('writing result to {}'.format(output_file))
	
	if method == 'canny_operator':
		save_img(ret[0], output_file)

		file_path, suffix = output_file.rsplit('.', 1)
		save_img(ret[1], '{}_direction_map.{}'.format(file_path, suffix))		
		return

	save_img(ret, output_file)

# Define the args to ease the calling of different operators
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_file', type=str, 
		help='source image file to process')
	parser.add_argument('--output_file', type=str, 
		help='image file to store the result.')
	parser.add_argument('--method', type=str, 
		help='the algorithm method to detect edge in the image')
	parser.add_argument('--sigma', type=float, default=None, 
		help='the hyperparameter for LoG and DoG')
	parser.add_argument('--sigma2', type=float, default=None, 
		help='the hyperparameter for DoG')
	parser.add_argument('--threshold', type=int, default=None, 
		help='the threshold used to binarize the image')

	args = parser.parse_args()
	main(args)


