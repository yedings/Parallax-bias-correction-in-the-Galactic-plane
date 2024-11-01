import numpy as np


def calc_pzpo(phot_g_mean_mag, colour, beta, filename):
    '''
    calculate parallax bias for five-parameter solutions located in |b|<=20 deg in Gaia DR3.
    
    Input parameter: 
        phot_g_mean_mag: G magnitude
        colour: effective wavenumber 
        beta: eclipic latitude 
        b: galactic latitude
        filename: path to the file with the coefficients
    
    Output parameter:
        parallax bias in microarcsecond
    '''
    sinBeta = np.sin(beta)
    # reads the file (.txt)
    input_file = np.genfromtxt(filename, delimiter=',')

    # auxiliary variables j and k
    j = list(map(int, input_file[0, 1:]))
    k = list(map(int, input_file[1, 1:]))
    # g vector
    g = input_file[2:, 0]
    # coefficients
    q_jk = input_file[2:, 1:]
    # shape
    n, m = q_jk.shape
    
    
    # basis functions evaluated at colour and ecl_lat
    c = [np.ones_like(colour),
         np.max((-0.24 * np.ones_like(colour), np.min((0.24 * np.ones_like(colour), colour - 1.48), axis=0)), axis=0),
         np.min((0.24 * np.ones_like(colour), np.max((np.zeros_like(colour), 1.48 - colour), axis=0)), axis=0) ** 3,
         np.min((np.zeros_like(colour), colour - 1.24), axis=0),
         np.max((np.zeros_like(colour), colour - 1.72), axis=0)]
    b = [np.ones_like(sinBeta), sinBeta, sinBeta ** 2 - 1. / 3]

    # coefficients must be interpolated between g(left) and g(left+1)
    # find the bin in g where gMag is
    ig = np.max((np.zeros_like(phot_g_mean_mag),
                 np.min((np.ones_like(phot_g_mean_mag) * (n - 2), np.digitize(phot_g_mean_mag, g, right=False) - 1),
                        axis=0)), axis=0).astype(int)

    # interpolate coefficients to gMag:
    h = np.max((np.zeros_like(phot_g_mean_mag),
                np.min((np.ones_like(phot_g_mean_mag), (phot_g_mean_mag - g[ig]) / (g[ig + 1] - g[ig])), axis=0)),
               axis=0)

    # sum over the product of the coefficients to get the zero-point
    pzpo = np.sum([((1 - h) * q_jk[ig, i] + h * q_jk[ig + 1, i]) * c[j[i]] * b[k[i]] for i in range(m)], axis=0)
    
    return pzpo


