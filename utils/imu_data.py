# This is a class to read the imu data from the specified file and return the requested parts of the data.

class IMUData:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_imu_data(self, start_line=None, end_line=None):
        """
        Read the imu data from the file. If start_line and end_line are not specified, the function will read the whole file.
        :param start_line: start line of the data to read
        :param end_line: end line of the data to read
        """
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = -1
        self.data = []
        with open(self.file_path, 'r') as f:
            for line in f.readlines()[start_line:end_line]:
                line = line.replace('\n', '')
                self.data.append(line)
        return self.data
    
    def get_attitude(self, start_line=None, end_line=None):
        """
        Get the yaw data from the imu file. if start_line and end_line are not specified, the function will read the whole file. if data is already loaded, it will not read the file again.
        :param start_line: start line of the data to read
        :param end_line: end line of the data to read
        """
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = -1

        if not hasattr(self, 'data'):
            self.read_imu_data(start_line, end_line)
        roll = []
        pitch = []
        yaw = []
        for line in self.data:
            line = line.split('|')
            attitude = line[2].split(',')
            r = float(attitude[0].split(': ')[1])
            p = float(attitude[1].split(': ')[1])
            y = float(attitude[2].split(': ')[1])
            roll.append(r)
            pitch.append(p)
            yaw.append(y)
        return roll, pitch, yaw
    
    def get_latlonalt_(self, start_line=None, end_line=None):
        """
        Get the latitude, longitude, altitude data from the imu file. if start_line and end_line are not specified, the function will read the whole file. if data is already loaded, it will not read the file again.
        :param start_line: start line of the data to read
        :param end_line: end line of the data to read
        """
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = -1
        if not hasattr(self, 'data'):
            self.read_imu_data(start_line, end_line)
        lat = []
        lon = []
        alt = []
        for line in self.data:
            cnt = line.count('|')
            if cnt != 5:
                # when the imu is just started, it doesn't have the lat, lon and alt data
                continue
            line = line.split('|')
            latlon = line[3].split(',')
            altt = line[4]
            la = float(latlon[0].split(': ')[1])
            lo = float(latlon[1].split(': ')[1])
            al = float(altt.split(': ')[1])
            lat.append(la)
            lon.append(lo)
            alt.append(al)
        return lat, lon, alt

    def get_utctimestamp(self, start_line=None, end_line=None):
        """     
        Get the timestamp data from the imu file. if start_line and end_line are not specified, the function will read the whole file. if data is already loaded, it will not read the file again.
        :param start_line: start line of the data to read
        :param end_line: end line of the data to read
        """
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = -1
        if not hasattr(self, 'data'):
            self.read_imu_data(start_line, end_line)
        timestamp = []
        for line in self.data:
            line = line.split('|')
            time = line[0].split(': ')[1]
            timestamp.append(time)
        return timestamp
    
    def get_latlonalt(self, start_line=None, end_line=None):
        """
        Get the latitude, longitude and altitude data from the imu file. if start_line and end_line are not specified, the function will read the whole file. if data is already loaded, it will not read the file again.
        Note: not all the imu files have the latitude, longitude and altitude data.
        :param start_line: start line of the data to read
        :param end_line: end line of the data to read
        """
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = -1
        if not hasattr(self, 'data'):
            self.read_imu_data(start_line, end_line)
        lat_lon_alt = []

        for line in self.data:
            line = line.split('|')
            if len(line) < 4:
                continue
            a = float(line[4].split(': ')[1])
            latlon = line[3].split(',')
            l = float(latlon[0].split(': ')[1])
            lo = float(latlon[1].split(': ')[1])
            lat_lon_alt.append([l, lo, a])
        return lat_lon_alt