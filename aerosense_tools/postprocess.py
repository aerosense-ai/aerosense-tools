import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks
from ahrs.filters import Madgwick, Mahony
from scipy.linalg import norm


class BladeIMU():
    """
    A class to process IMU data mounted on a windturbine blade.

    ...

    Attributes
    ----------
    time : numpy array of floats
        time of measurements in seconds starting from 0
    acc_mps2 : 3D numpy array of floats
        accelorometer values in m/s^2
    gyr_rps : 3D numpy array of floats
        gyroscope values in rad/s
    standstill : bool
        True if the blade is in standstill
    distance_from_tip : float
        distance from the tip of the blade to the IMU in meters
    blade_diameter : float
        diameter of the blade in meters
    hub_height : float
        height of the hub in meters
    N : int
        number of measurements
    radius : float
        radius of the blade in meters
    pitch : list of floats
        pitch angle of the blade in degrees
    azimuth : list of floats
        azimuth angle of the blade in degrees
    acc_peak_locations : 3D numpy array of floats
        locations of peaks in the accelerometer data
    acc_min_locations : 3D numpy array of floats
        locations of minima in the accelerometer data
    acc_peaks : 3D numpy array of floats
        interpolated function between peaks in the accelerometer data
    acc_mins : 3D numpy array of floats
        interpolated function between minima in the accelerometer data
    acc_centrip : 3D numpy array of floats
        centripetal force in the accelerometer data in all axes
    acc_g : 3D numpy array of floats
        gravity vector in the accelerometer data in all axes

    Methods
    -------
    update_acc()
        Decomposes accelerometer signal into centripetal force and gravity vector.
    update_peaks_mins(message=False)
        Peak detection helper function. Calls get_acc_peaks_mins for each axis.
    get_acc_peaks_mins(acc,time,N,distance=50,width=30,prominence=10)
        Peak detection helper function. Calls find_peaks from scipy.signal.
    calculate_precone()
        Calculates the precone angle of the blade.
    calculate_deflection()
        Estimates the flapwise deflections.
    calculate_pitch()
        Calculates the pitch angle of the blade.
    calculate_azimuth()
        Calculates the azimuth angle of the blade.
    get_height()
        Calculates the height of the blade.
    calibrate_gyro(bias)
        Calibrate the gyroscope by adding a bias
    calibrate_acc(bias)
        Calibrate the accelerometer by adding a bias
    rotate()
        Rotates the IMU's coordinates with given euler angles.
    blade_velocity()
        Calculates the blade velocity.
    AHRS_filter()
        Calculates the IMU's orientation to the inertial frame with AHRS filter library.
    q_as_euler()
        Converts the quaternion to euler angles.
    """

    def __init__(self, time, acc_mps2, gyr_rps, standstill=False, distance_from_tip=1.6, blade_diameter=12.8,
                 hub_height=18):
        self.time = time
        self.acc_mps2 = acc_mps2
        self.gyr_rps = gyr_rps
        self.standstill = standstill
        self.radius = blade_diameter / 2 - distance_from_tip
        self.hub_height = hub_height
        self.N = len(time)
        if self.N != len(acc_mps2[0]) or self.N != len(gyr_rps[0]):
            raise ValueError('Data arrays must be of same length')

        # Peak detection requires blade to be in motion.
        # gyr_rps[2] (z-axis) can be used to check for standstill as it presumably points closest to the direction of the rotational axis.
        threshold = 0.01
        if ((abs(self.gyr_rps[2]) < threshold).any() or standstill):
            self.standstill = True
            print('Standstill detected')
        else:
            print('Blade in motion')
            self.pitch = []
            self.azimuth = []
            self.acc_peak_locations = []
            self.acc_min_locations = []
            self.acc_peaks = []
            self.acc_mins = []
            self.acc_centrip = []
            self.acc_g = []
            self.update_acc()

    def update_acc(self):
        '''
        Decomposes accelerometer signal into centripetal force and gravity vector.
        '''
        self.update_peaks_mins(message=True)
        self.acc_centrip = ((self.acc_peaks + self.acc_mins) / 2.)
        self.acc_g = self.acc_mps2 - self.acc_centrip

    def update_peaks_mins(self, message=False):
        '''
        Peak detection helper function. Calls get_acc_peaks_mins for each axis.
        '''
        self.acc_peak_locations = []
        self.acc_min_locations = []
        self.acc_peaks = []
        self.acc_mins = []
        w_norm = np.mean((self.gyr_rps[0] ** 2 + self.gyr_rps[1] ** 2 + self.gyr_rps[2] ** 2) ** 0.5)
        distance = 50 * np.pi / w_norm

        axs = ['x', 'y', 'z']  # Just for printing out information
        for i in range(3):
            if i == 2:
                detect = self.get_acc_peaks_mins(self.acc_mps2[i], self.time, self.N, width=40)
            else:
                detect = self.get_acc_peaks_mins(self.acc_mps2[i], self.time, self.N, distance=distance, width=30,
                                                 prominence=10)
            # else : detect = self.get_acc_peaks_mins(self.acc_mps2[i],self.time,self.N,width=30)
            acc_peak_loc_i, acc_peaks_i, acc_min_loc_i, acc_mins_i = detect
            if message and len(acc_peak_loc_i) == 0: print("No peaks found for {}-axis".format(axs[i]))

            self.acc_peak_locations.append(acc_peak_loc_i)
            self.acc_min_locations.append(acc_min_loc_i)
            self.acc_peaks.append(acc_peaks_i)
            self.acc_mins.append(acc_mins_i)

        self.acc_peak_locations = np.asarray(self.acc_peak_locations, dtype='object')
        self.acc_min_locations = np.asarray(self.acc_min_locations, dtype='object')
        self.acc_peaks = np.asarray(self.acc_peaks, dtype='object')
        self.acc_mins = np.asarray(self.acc_mins, dtype='object')

    def get_acc_peaks_mins(self, signal, time, sample_len, height=None, threshold=None, distance=None, \
                           prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None):
        '''
        Get peak and minima locations of a signal, as well as an interpolated signal between the points.

        Parameters
        ----------
        signal: Nx1 np.ndarray
            Signal for which the peaks should be detected.
        time: Nx1 np.ndarray
            Timeline corresponding to the signal.

        Returns
        -------
        peak_locations: np.ndarray
            Detected peaks of the signal.
        peaks_interp: Nx1 np.ndarray
            Function interpolated between peak points.
        min_locations: np.ndarray
            Detected minima of the signal.
        mins_interp: Nx1 np.ndarray
            Function interpolated between min points.
        '''

        N = sample_len

        shape = signal.shape
        peaks_interp = np.empty(shape)
        mins_interp = np.empty(shape)

        # Find peak locations of ay
        peak_locations, _ = find_peaks(signal[0:N], height=height, threshold=threshold, distance=distance, \
                                       prominence=prominence, width=width, wlen=wlen, rel_height=rel_height,
                                       plateau_size=plateau_size)
        min_locations, _ = find_peaks(-signal[0:N], height=height, threshold=threshold, distance=distance, \
                                      prominence=prominence, width=width, wlen=wlen, rel_height=rel_height,
                                      plateau_size=plateau_size)

        if len(peak_locations) > 0 and len(min_locations) > 0:
            # interpolate peak points
            peaks_interp = np.interp(time, time[peak_locations], signal[peak_locations])
            mins_interp = np.interp(time, time[min_locations], signal[min_locations])
        else:
            peaks_interp = np.zeros(self.N)
            mins_interp = np.zeros(self.N)

        return peak_locations, peaks_interp, min_locations, mins_interp

    def calculate_precone(self):
        """
        Finds the precone angle of the blade (active rotation).
        Steps taken before calling this function:
        1. Rotate the IMU to the blade frame. (if not already done)
        2. Blade must be in standstill pointing directly downwards.
        3. Blade must be at 0 degree pitch.

        Returns
        -------
        precone:
            float - Precone angle of the blade.
        """
        if (self.standstill == False):
            raise ValueError('Blade must be in standstill')

        precone = (np.arctan2(self.acc_mps2[2],
                              self.acc_mps2[1]) * 180 / np.pi).mean()  # mean of precone angle in degrees
        return precone

    def calculate_deflection(self):
        """
        Estimates the flapwise deflections.
        Only delivers accurate results during fast rotational speeds.

        Returns
        -------
        deflections:
            float - Estimation of flapwise deflection
        """
        if (self.standstill == True):
            raise ValueError('Blade must be in motion')

        deflection = 180 - (np.arctan2(self.acc_mps2[2].mean(),
                                       self.acc_mps2[1].mean()) * 180 / np.pi)  # mean of deflection angle in degrees
        return deflection

    def calculate_pitch(self):
        """
        Finds the pitch angle of the blade.
        Rotation to 0 degree precone must be done before this function is called.
        Blade must be in motion.

        Returns
        -------
        pitch:
            1xN np.ndarray - Contains the pitch angle of the blade.
        """
        if (self.standstill == True):
            raise ValueError('Blade must be in motion')

        pitch = (np.arctan2(self.gyr_rps[0], -self.gyr_rps[2]) * 180 / np.pi)  # pitch angle in degrees
        self.pitch = pitch
        return pitch

    def calculate_azimuth(self):
        """
        Finds the azimuth angle of the blade.
        Rotation to 0 degree precone and 0 degree pitch must be done before this function is called.
        Blade must be in motion.

        Returns
        -------
        azimuth:
            1xN np.ndarray - Contains the azimuth angle of the blade.
        """
        if (self.standstill == True):
            raise ValueError('Blade must be in motion')
        self.update_acc()
        ag_x = -self.acc_g[0]
        ag_y = self.acc_g[1]

        azimuth = np.array(
            [np.arctan2(ag_x[i], ag_y[i]) * 180 / np.pi for i in range(self.N)])  # azimuth angle in degrees
        # change range from [180,-180] to [0,-360]
        azimuth = -azimuth - 180
        self.azimuth = azimuth
        return azimuth

    def get_height(self, azimuth):
        """
        Uses azimuthal angle to determine height of the sense relative to the ground.

        Inputs
        ------
        b_psi: Nx1 np.ndarray
            Azimuthal angle of blade.

        Returns
        -------
        height: Nx1 np.ndarray
            Height of sensor relative to the ground.
        """
        return self.hub_height - self.radius * np.sin(azimuth * np.pi / 180 - np.pi / 2)

    def calibrate_gyro(self, bias=np.array([[0], [0], [0]])):
        self.gyr_rps += bias

    def calibrate_acc(self, bias=np.array([[0], [0], [0]])):
        self.acc_mps2 += bias

    def rotate(self, angles):
        '''
        Rotates the IMU's coordinates with given euler angles.
        Rotation according to scipy.spatial.transform.Rotation library.

        Inputs
        ------
        Angles: 3xN np.ndarray
            Angles (x_rotation,y_rotation,z_rotation) in degrees, by which the coordinates shall be rotated.
        '''
        accT = np.transpose(self.acc_mps2)
        gyrT = np.transpose(self.gyr_rps)
        angles = np.transpose(angles)
        rot_gyrT = np.zeros(gyrT.shape)
        rot_accT = np.zeros(accT.shape)

        for i in range(self.N):
            rot = R.from_euler('xyz', angles[i], degrees=True)  # intrinsic
            rot_gyrT[i] = rot.apply(gyrT[i])
            rot_accT[i] = rot.apply(accT[i])

        self.acc_mps2 = np.transpose(rot_accT)
        self.gyr_rps = np.transpose(rot_gyrT)

        if (self.standstill == False):
            self.update_acc()  # Important!

    def blade_velocity(self):
        '''
        Makes use of the peaks and minima caused by oscillations due to the gravity vector to calculate the IMU's
        IMU angular velocity.
        Only useful if the IMU is in motion (otherwise no peaks are detected).

        Returns
        -------
        w_hat: Nx1 np.ndarray
            Estimated rotational velocity
        '''
        w_hat = np.tile(np.nan, self.N)
        w_hat_idx = np.concatenate(
            [self.acc_peak_locations[1], self.acc_min_locations[1]])  # Get peak locations from all axes (x,y,z)
        w_hat_idx = w_hat_idx.astype('int')
        for i in range(1, len(w_hat_idx)):
            w_hat[w_hat_idx[i]] = 1. / (self.time[w_hat_idx[i]] - self.time[w_hat_idx[i - 1]]) * 2 * np.pi  # w = 2*pi/T

        w_hat = np.interp(self.time, self.time[w_hat_idx], w_hat[w_hat_idx])
        return -w_hat  # return negative because of clockwise rotation

    def AHRS_filter(self, sample_rate, ahrs='Madgwick', euler=False, no_ac=True, q0=None, beta=0.033, k_P=1, k_I=0.3):
        '''
        Calculates the IMU's orientation using the Madgwick filter implementation from
        the AHRS.filters library. The orientation is calculated with the DC offset removed from accelerometer data.

        Parameters
        ----------
        sample_rate : int
            sample rate (necessary for madgwick/Mahony filter). Usually around 100
        ahrs:
            The desired ahrs, either 'Madgwick' or 'Mahony'
        euler: boolean
            If euler is True, the orientation is returned in Euler representation
        no_ac: boolean
            If no_ac is True, the orientation is calculated with the DC offset removed from accelerometer data.
        q0: 1x4 list
            Initial orientation of the IMU.
        beta: float
            Madgwick filter parameter
        k_P: float
            Mahony filter parameter
        k_I: float
            Mahony filter parameter

        Returns
        -------
        orientation:
            If euler:
                3xN np.ndarray - Euler angles of the orientation, following the Tait-Bryan convention (roll,pitch,yaw).
            else:
                Nx4 np.ndarray - Quaternion representation of the orientation.

        '''
        self.update_acc()
        w_norm = norm(self.gyr_rps, axis=0)
        if no_ac:
            w_threshold = 0.2
        else:
            w_threshold = 100
        acc_T = np.transpose(self.acc_mps2).astype('float')
        gyr_T = np.transpose(self.gyr_rps)
        acc_g_T = np.transpose(self.acc_g).astype('float')

        if ahrs == 'Madgwick':
            filter = Madgwick(beta=beta, frequency=sample_rate)
        elif ahrs == 'Mahony':
            filter = Mahony(k_P=k_P, k_I=k_I, frequency=sample_rate)

        Q = np.tile(q0, (self.N, 1))
        for t in range(1, self.N):
            if w_norm[t] > w_threshold:
                Q[t] = filter.updateIMU(Q[t - 1], gyr=gyr_T[t], acc=acc_g_T[t])
            else:
                Q[t] = filter.updateIMU(Q[t - 1], gyr=gyr_T[t], acc=acc_T[t])

        orientation = Q

        if euler:
            # Calculate orientations as euler angles
            orientation = self.q_as_euler(orientation)
            # change range from [180,-180] to [0,-360]
            orientation[2][orientation[2] > 0] -= 360
        return orientation

    def q_as_euler(self, Q):
        """
        Converts an array of Quaternions to their corresponding Euler angles.
        The conversion follows the Tait-Bryan (or intrinsic zyx / extrinsic XYZ) convention.
        Used by AHRS_filter().

        Parameters
        ----------
        Q: Nx4 np.ndarray
            Contains the Quaternions

        Returns
        -------
        angles: 3xN np.ndarray
            Contains the corresponding roll angle. -pi < phi < pi
            Contains the corresponding pitch angle. -pi/2 < theta < pi/2
            Contains the corresponding yaw angle. -pi < psi < pi
        """
        N = len(Q)
        angles = np.zeros((3, N))

        for i in range(N):
            # Using the Scipy Rotation library with the Tait-Bryan (roll-pitch-yaw) convention
            rot = R.from_quat(np.array([Q[i, 1], Q[i, 2], Q[i, 3], Q[i, 0]])).as_euler('XYZ', degrees=True)

            angles[0, i] = rot[0]
            angles[1, i] = rot[1]
            angles[2, i] = rot[2]

        return angles


class PostProcess:

    @staticmethod
    def processIMU(imu_signal):
        imu_data = imu_signal.dataframe
        processed_IMU = imu_data
        return processed_IMU

