import cv2
import numpy as np
import operator
import matplotlib.pyplot as plt

def hist_match_polynomial(source, template, polynomial_order=4):
    """
    Adjust the pixel values of one image so that its histogram matches another
    :param source: Image to transform (numpy array of size MxNx3)
    :param template: Image to match (numpy array, expected to be 3-channel but can be larger than source)
    """
    source = cv2.cvtColor(source,cv2.COLOR_GRAY2BGR)
    template = cv2.cvtColor(template,cv2.COLOR_GRAY2BGR)
    
    source_copy = np.copy(source)
    template_copy = np.copy(template)

    train_hist = np.zeros((3, 256), dtype='int')
    target_hist = np.zeros((3, 256), dtype='int')

    ''' Calculate histogram of source and target image '''
    for channel_ix in range(3):
        source = source_copy[:, :, channel_ix]
        template = template_copy[:, :, channel_ix]
        oldshape = source.shape

        # crop template to source shape
        h, w = oldshape[:2]
        template = crop_center(template, (h, w))

        train_hist[channel_ix, :] += count_pixel_values(source)
        target_hist[channel_ix, :] += count_pixel_values(template)

    # convert histograms back to values for polynomial fitting
    coefficients = {}
    for channel_ix, channel in enumerate(['polyR', 'polyG', 'polyB']):
        train_values = []
        target_values = []
        for value_ix in range(256):
            train_values += [value_ix] * train_hist[channel_ix, value_ix]
            target_values += [value_ix] * target_hist[channel_ix, value_ix]

        assert len(train_values) == len(target_values)

        coefficients[channel] = np.polyfit(train_values, target_values, polynomial_order)

    matched = color_transform_polynoms(source_copy, coefficients['polyR'], coefficients['polyG'], coefficients['polyB'])
    matched = cv2.cvtColor(matched,cv2.COLOR_BGR2GRAY)
    return matched

def count_pixel_values(ary):
    """
    Helper for counting pixel values in a 256-color image
    """
    counts, _ = np.histogram(ary, bins=range(257))
    return counts

# helper for cropping image center
def crop_center(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def color_transform_polynoms(img, polyR, polyG, polyB):
    """
    Performs a color transform on the rgb channels of a numpy array
    :param img: a numpy array, rgb color space
    :param polyR: polynom defining mapping on Red Channel, array of values see numpy.polynomial
    :param polyG: polynom defining mapping on Green Channel, array of values see numpy.polynomial
    :param polyB: polynom defining mapping on Blue Channel, array of values see numpy.polynomial
    :return: a numpy image array, RGB
    """

    mapRGB = np.zeros((3, 256))
    mapRGB[0] = np.polyval(polyR, np.arange(0, 256))
    mapRGB[1] = np.polyval(polyG, np.arange(0, 256))
    mapRGB[2] = np.polyval(polyB, np.arange(0, 256))
    mapRGB[mapRGB < 0] = 0
    mapRGB[mapRGB > 255] = 255

    new_img = img.copy()
    new_img[..., 0] = mapRGB[0][new_img[..., 0]]
    new_img[..., 1] = mapRGB[1][new_img[..., 1]]
    new_img[..., 2] = mapRGB[2][new_img[..., 2]]
    return np.uint8(new_img)

def hist_match_uniform(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """ 
    
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values = interp_t_values.astype(olddtype)
    matched = interp_t_values[bin_idx].reshape(oldshape)
 
    return matched

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

def show_CDF(source, template, matched):
    x1, y1 = ecdf(source.ravel())
    x2, y2 = ecdf(template.ravel())
    x3, y3 = ecdf(matched.ravel())
    
    fig = plt.figure()
    gs = plt.GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()
    
    ax1.imshow(source, cmap="gray")
    ax1.set_title('Source')
    ax2.imshow(template, cmap="gray")
    ax2.set_title('template')
    ax3.imshow(matched, cmap="gray")
    ax3.set_title('Matched')
    
    ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
    ax4.plot(x2, y2 * 100, '-k', lw=3, label='Template')
    ax4.plot(x3, y3 * 100, '--r', lw=3, label='Matched')
    ax4.set_xlim(x1[0], x1[-1])
    ax4.set_xlabel('Pixel value')
    ax4.set_ylabel('Cumulative %')
    ax4.legend(loc=5)
#    plt.savefig('matched_CDF.jpg')
    plt.show()

