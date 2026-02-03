"""
VASTControlClass - Python API for VAST Lite 1.5.0

This module provides a Python interface to the VAST (Volume Annotation and Segmentation Tool)
application via TCP socket communication. It is a complete port of the MATLAB VASTControlClass
version 5.14.

Key Features:
- Binary protocol communication with VAST application
- Complete API coverage (81 methods)
- RLE compression/decompression for efficient data transfer
- Support for segmentation, EM images, and screenshots
- Annotation object and skeleton operations
- Layer management and view control

Basic Usage:
    >>> from VCC import VASTControlClass
    >>>
    >>> # Connect to VAST
    >>> vcc = VASTControlClass(host='127.0.0.1', port=22081)
    >>> if vcc.connect(timeout=5):
    >>>     # Get volume info
    >>>     info, res = vcc.get_info()
    >>>     print(f"Data size: {info['datasizex']} x {info['datasizey']} x {info['datasizez']}")
    >>>
    >>>     # Retrieve segmentation image
    >>>     seg, res = vcc.get_seg_image_raw(0, 0, 99, 0, 99, 0, 0)
    >>>
    >>>     vcc.disconnect()

Version History:
    - 5.14 (2024-05-08): For VAST Lite 1.5.0
    - Python port: 2025-01-31

Author: Converted from MATLAB by Daniel Berger (Harvard-Lichtman)
"""

import socket
import struct
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


class VASTControlClass:
    """
    Python interface to VAST Lite 1.5.0 via TCP socket communication.

    This class implements the complete VAST API for controlling the VAST application,
    retrieving and setting segmentation data, managing layers, and working with
    annotation objects and skeletons.
    """

    # ============================================================================
    # VAST API MESSAGE ID CONSTANTS
    # ============================================================================

    GETINFO = 1
    GETNUMBEROFSEGMENTS = 2
    GETSEGMENTDATA = 3
    GETSEGMENTNAME = 4
    SETANCHORPOINT = 5
    SETSEGMENTNAME = 6
    SETSEGMENTCOLOR = 7
    GETVIEWCOORDINATES = 8
    GETVIEWZOOM = 9
    SETVIEWCOORDINATES = 10
    SETVIEWZOOM = 11
    GETNROFLAYERS = 12
    GETLAYERINFO = 13
    GETALLSEGMENTDATA = 14
    GETALLSEGMENTNAMES = 15
    SETSELECTEDSEGMENTNR = 16
    GETSELECTEDSEGMENTNR = 17
    SETSELECTEDLAYERNR = 18
    GETSELECTEDLAYERNR = 19
    GETSEGIMAGERAW = 20
    GETSEGIMAGERLE = 21
    GETSEGIMAGESURFRLE = 22
    SETSEGTRANSLATION = 23
    GETSEGIMAGERAWIMMEDIATE = 24
    GETSEGIMAGERLEIMMEDIATE = 25
    GETEMIMAGERAW = 30
    GETEMIMAGERAWIMMEDIATE = 31
    REFRESHLAYERREGION = 32
    GETPIXELVALUE = 33
    GETSCREENSHOTIMAGERAW = 40
    GETSCREENSHOTIMAGERLE = 41
    SETSEGIMAGERAW = 50
    SETSEGIMAGERLE = 51
    SETSEGMENTBBOX = 60
    GETFIRSTSEGMENTNR = 61
    GETHARDWAREINFO = 62
    ADDSEGMENT = 63
    MOVESEGMENT = 64
    GETDRAWINGPROPERTIES = 65
    SETDRAWINGPROPERTIES = 66
    GETFILLINGPROPERTIES = 67
    SETFILLINGPROPERTIES = 68

    GETANNOLAYERNROFOBJECTS = 70
    GETANNOLAYEROBJECTDATA = 71
    GETANNOLAYEROBJECTNAMES = 72
    ADDNEWANNOOBJECT = 73
    MOVEANNOOBJECT = 74
    REMOVEANNOOBJECT = 75
    SETSELECTEDANNOOBJECTNR = 76
    GETSELECTEDANNOOBJECTNR = 77
    GETAONODEDATA = 78
    GETAONODELABELS = 79

    SETSELECTEDAONODEBYDFSNR = 80
    SETSELECTEDAONODEBYCOORDS = 81
    GETSELECTEDAONODENR = 82
    ADDAONODE = 83
    MOVESELECTEDAONODE = 84
    REMOVESELECTEDAONODE = 85
    SWAPSELECTEDAONODECHILDREN = 86
    MAKESELECTEDAONODEROOT = 87

    SPLITSELECTEDSKELETON = 88
    WELDSKELETONS = 89

    GETANNOOBJECT = 90
    SETANNOOBJECT = 91
    ADDANNOOBJECT = 92
    GETCLOSESTAONODEBYCOORDS = 93
    GETAONODEPARAMS = 94
    SETAONODEPARAMS = 95

    GETAPIVERSION = 100
    GETAPILAYERSENABLED = 101
    SETAPILAYERSENABLED = 102
    GETSELECTEDAPILAYERNR = 103
    SETSELECTEDAPILAYERNR = 104

    GETCURRENTUISTATE = 110
    GETERRORPOPUPSENABLED = 112
    SETERRORPOPUPSENABLED = 113
    SETUIMODE = 114
    SHOWWINDOW = 115
    SET2DVIEWORIENTATION = 116
    GETPOPUPSENABLED = 117
    SETPOPUPSENABLED = 118

    ADDNEWLAYER = 120
    LOADLAYER = 121
    SAVELAYER = 122
    REMOVELAYER = 123
    MOVELAYER = 124
    SETLAYERINFO = 125
    GETMIPMAPSCALEFACTORS = 126

    EXECUTEFILL = 131
    EXECUTELIMITEDFILL = 132
    EXECUTECANVASPAINTSTROKE = 133
    EXECUTESTARTAUTOSKELETONIZATION = 134
    EXECUTESTOPAUTOSKELETONIZATION = 135
    EXECUTEISAUTOSKELETONIZATIONDONE = 136

    SETTOOLPARAMETERS = 151

    # ============================================================================
    # INITIALIZATION
    # ============================================================================

    def __init__(self, host: str = '127.0.0.1', port: int = 22081):
        """
        Initialize VAST Control Class.

        Args:
            host: VAST server hostname or IP address (default: '127.0.0.1')
            port: VAST server port (default: 22081)
        """
        self.host = host
        self.port = port
        self.client: Optional[socket.socket] = None
        self.is_connected = False

        # Version tracking
        self.this_version_nr = 5  # Version 5 for VAST Lite 1.3, 1.4, 1.5
        self.this_subversion_nr = 14

        # Response parsing state
        self.indata = b''
        self.in_res: Optional[int] = None
        self.in_int_data = np.array([], dtype=np.int32)
        self.in_uint_data = np.array([], dtype=np.uint32)
        self.in_double_data = np.array([], dtype=np.float64)
        self.in_char_data = ''
        self.in_text_data = []
        self.in_uint64_data = np.array([], dtype=np.uint64)

        # Error tracking
        self.last_error = 0
        self.parse_header_ok = False
        self.parse_header_len = 0

    # ============================================================================
    # CONNECTION MANAGEMENT
    # ============================================================================

    def connect(self, timeout: int = 100) -> int:
        """
        Establish TCP connection to VAST server.

        Args:
            timeout: Connection timeout in seconds (default: 100)

        Returns:
            1 if successful, 0 if failed
        """
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(timeout)
            self.client.connect((self.host, self.port))
            self.is_connected = True
            return 1
        except Exception as e:
            print(f"Connection failed: {e}")
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
            self.client = None
            self.is_connected = False
            self.last_error = 1
            return 0

    def disconnect(self) -> int:
        """
        Close connection to VAST server.

        Returns:
            1 if successful, 0 if failed
        """
        try:
            if self.client:
                self.client.close()
            self.is_connected = False
            return 1
        except Exception as e:
            print(f"Disconnect failed: {e}")
            return 0

    def get_last_error(self) -> int:
        """
        Get the last error code.

        Returns:
            Error code (0 = no error)
        """
        return self.last_error

    def get_control_class_version(self) -> Tuple[int, int, int]:
        """
        Get the version of this control class.

        Returns:
            Tuple[int, int, int]: (version, subversion, result_code)
        """
        return self.this_version_nr, self.this_subversion_nr, 1

    # ============================================================================
    # LOW-LEVEL COMMUNICATION
    # ============================================================================

    def _send_message(self, message_nr: int, message: bytes) -> None:
        """
        Send a binary message to VAST server.

        Message format:
        [0:4]    'VAST' header
        [4:12]   uint64 payload length (little-endian)
        [12:16]  uint32 message ID (little-endian)
        [16:]    payload data

        Args:
            message_nr: Message ID constant
            message: Binary payload data
        """
        # Calculate length (message data + 4 bytes for message number)
        length = len(message) + 4

        # Build header
        header = b'VAST'
        header += struct.pack('<Q', length)  # uint64 length
        header += struct.pack('<I', message_nr)  # uint32 message number

        # Send complete message
        full_message = header + message
        self.client.sendall(full_message)

    def _read_data_block(self) -> None:
        """
        Read data block from VAST server.

        Blocks until data is available, then stores in self.indata.
        """
        # Read header first (16 bytes)
        header = self._recv_exact(16)

        if len(header) < 16:
            self.indata = b''
            return

        # Parse length from header
        length = struct.unpack('<Q', header[4:12])[0]

        # Read payload
        payload = self._recv_exact(length)

        # Store complete message
        self.indata = header + payload

    def _recv_exact(self, num_bytes: int) -> bytes:
        """
        Receive exact number of bytes from socket.

        Args:
            num_bytes: Number of bytes to receive

        Returns:
            Bytes received
        """
        data = bytearray()
        while len(data) < num_bytes:
            chunk = self.client.recv(num_bytes - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def _parse_header(self, indata: bytes) -> None:
        """
        Parse VAST response header.

        Args:
            indata: Raw binary data from server
        """
        self.parse_header_ok = False
        self.parse_header_len = 0
        self.in_res = None

        if len(indata) < 16:
            return

        # Check magic header
        if indata[0:4] != b'VAST':
            return

        self.parse_header_ok = True

        # Parse length
        self.parse_header_len = struct.unpack('<Q', indata[4:12])[0]

        # Parse result code
        self.in_res = struct.unpack('<i', indata[12:16])[0]

    def _parse(self, indata: bytes) -> None:
        """
        Parse VAST response payload.

        Extracts type-tagged data from binary response:
        - Tag 1: uint32 (4 bytes)
        - Tag 2: double (8 bytes)
        - Tag 3: null-terminated text
        - Tag 4: int32 (4 bytes)
        - Tag 6: uint64 (8 bytes)

        Args:
            indata: Raw binary data from server
        """
        # Reset all data arrays
        self.in_int_data = []
        self.in_uint_data = []
        self.in_double_data = []
        self.in_char_data = ''
        self.in_text_data = []
        self.in_uint64_data = []
        self.in_res = None

        if len(indata) < 12:
            return

        # Check magic header
        if indata[0:4] != b'VAST':
            return

        # Parse length
        length = struct.unpack('<Q', indata[4:12])[0]

        if length != len(indata) - 12:
            return

        # Parse result code
        self.in_res = struct.unpack('<i', indata[12:16])[0]

        # Parse tagged data
        p = 16
        while p < len(indata):
            tag = indata[p]

            if tag == 1:  # uint32
                if p + 5 <= len(indata):
                    value = struct.unpack('<I', indata[p+1:p+5])[0]
                    self.in_uint_data.append(value)
                    p += 5
                else:
                    break

            elif tag == 2:  # double
                if p + 9 <= len(indata):
                    value = struct.unpack('<d', indata[p+1:p+9])[0]
                    self.in_double_data.append(value)
                    p += 9
                else:
                    break

            elif tag == 3:  # null-terminated text
                q = p + 1
                while q < len(indata) and indata[q] != 0:
                    q += 1

                if q < len(indata) and indata[q] == 0:
                    text = indata[p+1:q].decode('utf-8', errors='replace')
                    self.in_char_data = text
                    self.in_text_data.append(text)
                    p = q + 1
                else:
                    break

            elif tag == 4:  # int32
                if p + 5 <= len(indata):
                    value = struct.unpack('<i', indata[p+1:p+5])[0]
                    self.in_int_data.append(value)
                    p += 5
                else:
                    break

            elif tag == 6:  # uint64
                if p + 9 <= len(indata):
                    value = struct.unpack('<Q', indata[p+1:p+9])[0]
                    self.in_uint64_data.append(value)
                    p += 9
                else:
                    break
            else:
                # Unknown tag, stop parsing
                break

        # Convert lists to numpy arrays
        self.in_int_data = np.array(self.in_int_data, dtype=np.int32)
        self.in_uint_data = np.array(self.in_uint_data, dtype=np.uint32)
        self.in_double_data = np.array(self.in_double_data, dtype=np.float64)
        self.in_uint64_data = np.array(self.in_uint64_data, dtype=np.uint64)

    def _process_error(self) -> int:
        """
        Process error code from VAST response.

        Returns:
            1 if successful (inres == 0), 0 if error occurred
        """
        self.last_error = 0

        if self.in_res is None:
            self.last_error = 2  # Unexpected data
            return 0

        if self.in_res == 0:
            return 1  # Success
        else:
            # Error occurred
            if len(self.in_uint_data) >= 1:
                self.last_error = int(self.in_uint_data[0])
            else:
                self.last_error = 1  # Unknown error
            return 0

    # ============================================================================
    # BINARY ENCODING METHODS
    # ============================================================================

    @staticmethod
    def _bytes_from_int32(value: np.ndarray) -> bytes:
        """
        Encode int32 value(s) to binary format.

        Format: Tag 4 + 4-byte little-endian int32

        Args:
            value: int32 value or array

        Returns:
            Binary encoded data
        """
        val = np.atleast_1d(np.int32(value))
        result = bytearray()
        for v in val:
            result.extend(b'\x04')  # Tag 4
            result.extend(struct.pack('<i', v))
        return bytes(result)

    @staticmethod
    def _bytes_from_uint32(value: np.ndarray) -> bytes:
        """
        Encode uint32 value(s) to binary format.

        Format: Tag 1 + 4-byte little-endian uint32

        Args:
            value: uint32 value or array

        Returns:
            Binary encoded data
        """
        val = np.atleast_1d(np.uint32(value))
        result = bytearray()
        for v in val:
            result.extend(b'\x01')  # Tag 1
            result.extend(struct.pack('<I', v))
        return bytes(result)

    @staticmethod
    def _bytes_from_double(value: np.ndarray) -> bytes:
        """
        Encode double value(s) to binary format.

        Format: Tag 2 + 8-byte little-endian double

        Args:
            value: double value or array

        Returns:
            Binary encoded data
        """
        val = np.atleast_1d(np.float64(value))
        result = bytearray()
        for v in val:
            result.extend(b'\x02')  # Tag 2
            result.extend(struct.pack('<d', v))
        return bytes(result)

    @staticmethod
    def _bytes_from_text(value: str) -> bytes:
        """
        Encode text string to binary format.

        Format: Tag 3 + UTF-8 bytes + null terminator

        Args:
            value: Text string

        Returns:
            Binary encoded data
        """
        result = bytearray()
        result.extend(b'\x03')  # Tag 3
        result.extend(value.encode('utf-8'))
        result.extend(b'\x00')  # Null terminator
        return bytes(result)

    @staticmethod
    def _bytes_from_data(value: np.ndarray) -> bytes:
        """
        Encode raw data array to binary format.

        Format: Tag 5 + uint32(length) + raw bytes

        Args:
            value: Data array

        Returns:
            Binary encoded data
        """
        val = np.asarray(value).flatten()
        raw_bytes = val.tobytes()
        length = len(raw_bytes)

        result = bytearray()
        result.extend(b'\x05')  # Tag 5
        result.extend(struct.pack('<I', length))
        result.extend(raw_bytes)
        return bytes(result)

    # ============================================================================
    # STRUCT ENCODING/DECODING
    # ============================================================================

    def _decode_to_struct(self, indata: bytes) -> Tuple[Dict, int]:
        """
        Decode binary data to structured dictionary.

        Type codes:
        0: uint32, 1: int32, 2: uint64, 3: double, 4: string
        5: uint32 matrix, 6: int32 matrix, 7: uint64 matrix, 8: double matrix, 9: string array

        Args:
            indata: Binary data to decode

        Returns:
            Tuple: (decoded_dict, result_code)
                - decoded_dict: Dictionary with variable names as keys
                - result_code: 1 if success, 0 if failed
        """
        out = {}
        ptr = 0
        data_len = len(indata)

        try:
            while ptr < data_len:
                # Read variable name (null-terminated)
                ptr2 = ptr
                while ptr2 < data_len and indata[ptr2] != 0:
                    ptr2 += 1

                if ptr2 >= data_len:
                    break

                variable_name = indata[ptr:ptr2].decode('utf-8', errors='replace')
                ptr = ptr2 + 1

                if ptr >= data_len:
                    break

                # Read type
                data_type = indata[ptr]
                ptr += 1

                if data_type == 0:  # uint32
                    if ptr + 4 > data_len:
                        break
                    value = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    out[variable_name] = int(value)
                    ptr += 4

                elif data_type == 1:  # int32
                    if ptr + 4 > data_len:
                        break
                    value = struct.unpack('<i', indata[ptr:ptr+4])[0]
                    out[variable_name] = int(value)
                    ptr += 4

                elif data_type == 2:  # uint64
                    if ptr + 8 > data_len:
                        break
                    value = struct.unpack('<Q', indata[ptr:ptr+8])[0]
                    out[variable_name] = int(value)
                    ptr += 8

                elif data_type == 3:  # double
                    if ptr + 8 > data_len:
                        break
                    value = struct.unpack('<d', indata[ptr:ptr+8])[0]
                    out[variable_name] = float(value)
                    ptr += 8

                elif data_type == 4:  # string
                    ptr2 = ptr
                    while ptr2 < data_len and indata[ptr2] != 0:
                        ptr2 += 1

                    if ptr2 >= data_len:
                        break

                    value = indata[ptr:ptr2].decode('utf-8', errors='replace')
                    out[variable_name] = value
                    ptr = ptr2 + 1

                elif data_type == 5:  # uint32 matrix
                    if ptr + 8 > data_len:
                        break
                    xsize = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4
                    ysize = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4

                    matrix_bytes = xsize * ysize * 4
                    if ptr + matrix_bytes > data_len:
                        break

                    mtx_data = struct.unpack(f'<{xsize*ysize}I', indata[ptr:ptr+matrix_bytes])
                    mtx = np.array(mtx_data, dtype=np.uint32).reshape((ysize, xsize))
                    out[variable_name] = mtx
                    ptr += matrix_bytes

                elif data_type == 6:  # int32 matrix
                    if ptr + 8 > data_len:
                        break
                    xsize = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4
                    ysize = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4

                    matrix_bytes = xsize * ysize * 4
                    if ptr + matrix_bytes > data_len:
                        break

                    mtx_data = struct.unpack(f'<{xsize*ysize}i', indata[ptr:ptr+matrix_bytes])
                    mtx = np.array(mtx_data, dtype=np.int32).reshape((ysize, xsize))
                    out[variable_name] = mtx
                    ptr += matrix_bytes

                elif data_type == 7:  # uint64 matrix
                    if ptr + 8 > data_len:
                        break
                    xsize = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4
                    ysize = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4

                    matrix_bytes = xsize * ysize * 8
                    if ptr + matrix_bytes > data_len:
                        break

                    mtx_data = struct.unpack(f'<{xsize*ysize}Q', indata[ptr:ptr+matrix_bytes])
                    mtx = np.array(mtx_data, dtype=np.uint64).reshape((ysize, xsize))
                    out[variable_name] = mtx
                    ptr += matrix_bytes

                elif data_type == 8:  # double matrix
                    if ptr + 8 > data_len:
                        break
                    xsize = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4
                    ysize = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4

                    matrix_bytes = xsize * ysize * 8
                    if ptr + matrix_bytes > data_len:
                        break

                    mtx_data = struct.unpack(f'<{xsize*ysize}d', indata[ptr:ptr+matrix_bytes])
                    mtx = np.array(mtx_data, dtype=np.float64).reshape((ysize, xsize))
                    out[variable_name] = mtx
                    ptr += matrix_bytes

                elif data_type == 9:  # string array
                    if ptr + 4 > data_len:
                        break
                    nr_of_strings = struct.unpack('<I', indata[ptr:ptr+4])[0]
                    ptr += 4

                    string_array = []
                    for _ in range(nr_of_strings):
                        ptr2 = ptr
                        while ptr2 < data_len and indata[ptr2] != 0:
                            ptr2 += 1

                        if ptr2 >= data_len:
                            break

                        string_val = indata[ptr:ptr2].decode('utf-8', errors='replace')
                        string_array.append(string_val)
                        ptr = ptr2 + 1

                    out[variable_name] = string_array

                else:
                    # Unknown type
                    break

            return out, 1

        except Exception:
            return {}, 0

    def _encode_from_struct(self, indata: Dict) -> Tuple[bytes, int]:
        """
        Encode dictionary to binary format.

        Args:
            indata: Dictionary to encode

        Returns:
            Tuple: (encoded_bytes, result_code)
                - encoded_bytes: Binary encoded data
                - result_code: 1 if success, 0 if failed
        """
        result = bytearray()

        try:
            for key, value in indata.items():
                # Encode variable name
                result.extend(key.encode('utf-8'))
                result.extend(b'\x00')

                # Encode value based on type
                if isinstance(value, str):
                    # Type 4: string
                    result.extend(b'\x04')
                    result.extend(value.encode('utf-8'))
                    result.extend(b'\x00')

                elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                    # Type 9: string array
                    result.extend(b'\x09')
                    result.extend(struct.pack('<I', len(value)))
                    for s in value:
                        result.extend(s.encode('utf-8'))
                        result.extend(b'\x00')

                elif isinstance(value, np.ndarray):
                    if value.dtype == np.uint32:
                        if value.ndim == 1:
                            # Type 0: uint32 scalar or array
                            for v in value.flat:
                                result.extend(b'\x00')
                                result.extend(struct.pack('<I', v))
                        else:
                            # Type 5: uint32 matrix
                            result.extend(b'\x05')
                            result.extend(struct.pack('<I', value.shape[1]))  # xsize
                            result.extend(struct.pack('<I', value.shape[0]))  # ysize
                            result.extend(value.tobytes())

                    elif value.dtype == np.int32:
                        if value.ndim == 1:
                            # Type 1: int32 scalar or array
                            for v in value.flat:
                                result.extend(b'\x01')
                                result.extend(struct.pack('<i', v))
                        else:
                            # Type 6: int32 matrix
                            result.extend(b'\x06')
                            result.extend(struct.pack('<I', value.shape[1]))
                            result.extend(struct.pack('<I', value.shape[0]))
                            result.extend(value.tobytes())

                    elif value.dtype == np.uint64:
                        if value.ndim == 1:
                            # Type 2: uint64 scalar or array
                            for v in value.flat:
                                result.extend(b'\x02')
                                result.extend(struct.pack('<Q', v))
                        else:
                            # Type 7: uint64 matrix
                            result.extend(b'\x07')
                            result.extend(struct.pack('<I', value.shape[1]))
                            result.extend(struct.pack('<I', value.shape[0]))
                            result.extend(value.tobytes())

                    elif value.dtype == np.float64:
                        if value.ndim == 1:
                            # Type 3: double scalar or array
                            for v in value.flat:
                                result.extend(b'\x03')
                                result.extend(struct.pack('<d', v))
                        else:
                            # Type 8: double matrix
                            result.extend(b'\x08')
                            result.extend(struct.pack('<I', value.shape[1]))
                            result.extend(struct.pack('<I', value.shape[0]))
                            result.extend(value.tobytes())

                elif isinstance(value, int):
                    # Type 0: uint32 (default for Python int)
                    if value < 0:
                        result.extend(b'\x01')
                        result.extend(struct.pack('<i', value))
                    else:
                        result.extend(b'\x00')
                        result.extend(struct.pack('<I', value))

                elif isinstance(value, float):
                    # Type 3: double
                    result.extend(b'\x03')
                    result.extend(struct.pack('<d', value))

            return bytes(result), 1

        except Exception:
            return b'', 0

    # ============================================================================
    # INFO & METADATA
    # ============================================================================

    def get_info(self) -> Tuple[Dict, int]:
        """
        Get volume information.

        Returns:
            Tuple: (info_dict, result_code)
                - info_dict: Contains datasizex/y/z, voxelsizex/y/z, cubesizex/y/z, currentviewx/y/z, nrofmiplevels
                - result_code: 1 if success, 0 if failed
        """
        self._send_message(self.GETINFO, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) == 7 and len(self.in_double_data) == 3 and len(self.in_int_data) == 3:
                info = {
                    'datasizex': int(self.in_uint_data[0]),
                    'datasizey': int(self.in_uint_data[1]),
                    'datasizez': int(self.in_uint_data[2]),
                    'voxelsizex': float(self.in_double_data[0]),
                    'voxelsizey': float(self.in_double_data[1]),
                    'voxelsizez': float(self.in_double_data[2]),
                    'cubesizex': int(self.in_uint_data[3]),
                    'cubesizey': int(self.in_uint_data[4]),
                    'cubesizez': int(self.in_uint_data[5]),
                    'currentviewx': int(self.in_int_data[0]),
                    'currentviewy': int(self.in_int_data[1]),
                    'currentviewz': int(self.in_int_data[2]),
                    'nrofmiplevels': int(self.in_uint_data[6]),
                }
                return info, res
            else:
                self.last_error = 2
                return {}, 0
        else:
            return {}, res

    def get_api_version(self) -> Tuple[int, int, int]:
        """
        Get VAST API version.

        Returns:
            Tuple: (version, subversion, result_code)
        """
        self._send_message(self.GETAPIVERSION, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 2:
                version = int(self.in_uint_data[0])
                subversion = int(self.in_uint_data[1])
                return version, subversion, res
            else:
                self.last_error = 2
                return 0, 0, 0
        else:
            return 0, 0, res

    def get_hardware_info(self) -> Tuple[Dict, int]:
        """
        Get hardware information.

        Returns:
            Tuple: (hardware_dict, result_code)
                - hardware_dict: Contains graphics card and system info
                - result_code: 1 if success, 0 if failed
        """
        self._send_message(self.GETHARDWAREINFO, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_text_data) > 0:
                hardware = {'info': self.in_text_data[0]}
                return hardware, res
            else:
                return {}, res
        else:
            return {}, res

    def get_number_of_segments(self) -> Tuple[int, int]:
        """
        Get number of segments.

        Returns:
            Tuple: (num_segments, result_code)
        """
        self._send_message(self.GETNUMBEROFSEGMENTS, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 1:
                num_segments = int(self.in_uint_data[0])
                return num_segments, res
            else:
                self.last_error = 2
                return 0, 0
        else:
            return 0, res

    def get_number_of_layers(self) -> Tuple[int, int]:
        """
        Get number of layers.

        Returns:
            Tuple: (num_layers, result_code)
        """
        self._send_message(self.GETNROFLAYERS, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 1:
                num_layers = int(self.in_uint_data[0])
                return num_layers, res
            else:
                self.last_error = 2
                return 0, 0
        else:
            return 0, res

    def get_data_size_at_mip(self, layer_nr: int, mip: int) -> Tuple[int, int, int]:
        """
        Get data size at specific MIP level.

        Args:
            layer_nr: Layer number
            mip: MIP level

        Returns:
            Tuple: (sizex, sizey, sizez)
        """
        info, res = self.get_info()
        if res == 0:
            return 0, 0, 0

        mipmap_factors, res = self.get_mipmap_scale_factors(layer_nr)
        if res == 0:
            return 0, 0, 0

        if mip >= len(mipmap_factors):
            return 0, 0, 0

        factor = mipmap_factors[mip]
        sizex = int(np.ceil(info['datasizex'] / factor))
        sizey = int(np.ceil(info['datasizey'] / factor))
        sizez = int(np.ceil(info['datasizez'] / factor))

        return sizex, sizey, sizez

    # ============================================================================
    # SEGMENT OPERATIONS
    # ============================================================================

    def get_segment_data(self, seg_id: int) -> Tuple[Dict, int]:
        """
        Get data for a specific segment.

        Args:
            seg_id: Segment ID

        Returns:
            Tuple: (segment_dict, result_code)
                - segment_dict: Contains id, flags, col1, col2, anchorpoint, hierarchy, collapsednr, boundingbox
                - result_code: 1 if success, 0 if failed
        """
        message = self._bytes_from_uint32(seg_id)
        self._send_message(self.GETSEGMENTDATA, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_int_data) == 9 and len(self.in_uint_data) == 9:
                data = {
                    'id': int(self.in_uint_data[0]),
                    'flags': int(self.in_uint_data[1]),
                    'col1': int(self.in_uint_data[2]),
                    'col2': int(self.in_uint_data[3]),
                    'anchorpoint': self.in_int_data[0:3].tolist(),
                    'hierarchy': self.in_uint_data[4:8].tolist(),
                    'collapsednr': int(self.in_uint_data[8]),
                    'boundingbox': self.in_int_data[3:9].tolist(),
                }
                return data, res
            else:
                self.last_error = 2
                return {}, 0
        else:
            return {}, res

    def get_segment_name(self, seg_id: int) -> Tuple[str, int]:
        """
        Get name of a segment.

        Args:
            seg_id: Segment ID

        Returns:
            Tuple: (name, result_code)
        """
        message = self._bytes_from_uint32(seg_id)
        self._send_message(self.GETSEGMENTNAME, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            name = self.in_char_data
            return name, res
        else:
            return '', res

    def get_all_segment_data(self) -> Tuple[List[Dict], int]:
        """
        Get data for all segments.

        Returns:
            Tuple: (segment_list, result_code)
                - segment_list: List of segment dictionaries
                - result_code: 1 if success, 0 if failed
        """
        self._send_message(self.GETALLSEGMENTDATA, b'')
        self._read_data_block()
        self._parse_header(self.indata)

        expected_length = self.parse_header_len + 12

        # Read all data if needed
        if len(self.indata) < expected_length:
            additional_bytes = expected_length - len(self.indata)
            more_data = self._recv_exact(additional_bytes)
            self.indata = self.indata + more_data

        if self.in_res == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            # Parse binary segment data
            uid = np.frombuffer(self.indata[16:], dtype=np.uint32)
            sid = np.frombuffer(self.indata[16:], dtype=np.int32)

            num_segments = uid[0]
            segdata = []
            sp = 1  # Start position after count

            for i in range(num_segments):
                seg = {
                    'id': i,
                    'flags': int(uid[sp]),
                    'col1': int(uid[sp+1]),
                    'col2': int(uid[sp+2]),
                    'anchorpoint': [int(sid[sp+3]), int(sid[sp+4]), int(sid[sp+5])],
                    'hierarchy': [int(uid[sp+6]), int(uid[sp+7]), int(uid[sp+8]), int(uid[sp+9])],
                    'collapsednr': int(uid[sp+10]),
                    'boundingbox': [int(sid[sp+11]), int(sid[sp+12]), int(sid[sp+13]),
                                   int(sid[sp+14]), int(sid[sp+15]), int(sid[sp+16])],
                }
                segdata.append(seg)
                sp += 17

            return segdata, res
        else:
            return [], res

    def get_all_segment_data_matrix(self) -> Tuple[np.ndarray, int]:
        """
        Get all segment data as matrix matching MATLAB's format.

        Returns:
            Tuple: (segment_matrix, result_code)
                - segment_matrix: Nx24 array of segment data matching MATLAB format:
                    col 0: segment ID (0-based index)
                    col 1: flags
                    col 2-5: color1 RGBA (4 bytes unpacked)
                    col 6-9: color2 RGBA (4 bytes unpacked)
                    col 10-12: anchorpoint (x, y, z)
                    col 13-16: hierarchy (4 values: parent, firstchild, prevsib, nextsib)
                    col 17: collapsed number
                    col 18-23: bounding box (6 values)
                - result_code: 1 if success, 0 if failed
        """
        # Get raw data directly like MATLAB does
        self._send_message(self.GETALLSEGMENTDATA, b'')
        self._read_data_block()
        self._parse_header(self.indata)

        expected_length = self.parse_header_len + 12

        # Read all data if needed
        if len(self.indata) < expected_length:
            additional_bytes = expected_length - len(self.indata)
            more_data = self._recv_exact(additional_bytes)
            self.indata = self.indata + more_data

        if self.in_res == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res != 1:
            return np.array([]), res

        # Parse binary segment data - match MATLAB exactly
        uid = np.frombuffer(self.indata[16:], dtype=np.uint32)
        sid = np.frombuffer(self.indata[16:], dtype=np.int32)

        num_segments = int(uid[0])
        if num_segments == 0:
            return np.array([]), res

        # Create 24-column matrix like MATLAB
        # MATLAB: segdatamatrix=zeros(uid(1),24);
        matrix = np.zeros((num_segments, 24), dtype=np.float64)
        sp = 1  # Start position (MATLAB uses sp=2 because 1-indexed)

        for i in range(num_segments):
            # MATLAB: segdatamatrix(i,1)=i-1;  (but then removes first row, so effectively i)
            matrix[i, 0] = i  # segment ID (0-based, will match after removing background)
            
            # MATLAB: segdatamatrix(i,2)=uid(sp);
            matrix[i, 1] = uid[sp]  # flags
            
            # MATLAB: segdatamatrix(i,3:6) - color1 RGBA unpacked from uint32
            col1 = uid[sp + 1]
            matrix[i, 2] = (col1 >> 24) & 255  # R
            matrix[i, 3] = (col1 >> 16) & 255  # G
            matrix[i, 4] = (col1 >> 8) & 255   # B
            matrix[i, 5] = col1 & 255          # A
            
            # MATLAB: segdatamatrix(i,7:10) - color2 RGBA unpacked from uint32
            col2 = uid[sp + 2]
            matrix[i, 6] = (col2 >> 24) & 255  # R
            matrix[i, 7] = (col2 >> 16) & 255  # G
            matrix[i, 8] = (col2 >> 8) & 255   # B
            matrix[i, 9] = col2 & 255          # A
            
            # MATLAB: segdatamatrix(i,11:13)=id(sp+3:sp+5); - anchorpoint (signed)
            matrix[i, 10] = sid[sp + 3]
            matrix[i, 11] = sid[sp + 4]
            matrix[i, 12] = sid[sp + 5]
            
            # MATLAB: segdatamatrix(i,14:17)=uid(sp+6:sp+9); - hierarchy (unsigned)
            # hierarchy[0] = parent ID (this is column 14 in MATLAB 1-indexed = column 13 in 0-indexed)
            matrix[i, 13] = uid[sp + 6]  # parent
            matrix[i, 14] = uid[sp + 7]  # first child
            matrix[i, 15] = uid[sp + 8]  # prev sibling
            matrix[i, 16] = uid[sp + 9]  # next sibling
            
            # MATLAB: segdatamatrix(i,18)=uid(sp+10); - collapsed number
            matrix[i, 17] = uid[sp + 10]
            
            # MATLAB: segdatamatrix(i,19:24)=id(sp+11:sp+16); - bounding box (signed)
            matrix[i, 18] = sid[sp + 11]
            matrix[i, 19] = sid[sp + 12]
            matrix[i, 20] = sid[sp + 13]
            matrix[i, 21] = sid[sp + 14]
            matrix[i, 22] = sid[sp + 15]
            matrix[i, 23] = sid[sp + 16]
            
            sp += 17

        # MATLAB: segdatamatrix=segdatamatrix(2:end,:); - remove first row (background)
        if num_segments > 1:
            matrix = matrix[1:, :]
            # Update IDs to match (they should already be correct since we used i as ID)
        
        return matrix, res

    def get_all_segment_names(self) -> Tuple[List[str], int]:
        """
        Get names of all segments.

        Returns:
            Tuple: (name_list, result_code)
        """
        self._send_message(self.GETALLSEGMENTNAMES, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            names = self.in_text_data
            return names, res
        else:
            return [], res

    def set_segment_name(self, seg_id: int, name: str) -> int:
        """
        Set segment name.

        Args:
            seg_id: Segment ID
            name: New name

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(seg_id) + self._bytes_from_text(name)
        self._send_message(self.SETSEGMENTNAME, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_segment_color8(self, seg_id: int, r1: int, g1: int, b1: int, p1: int,
                          r2: int, g2: int, b2: int, p2: int) -> int:
        """
        Set segment color using 8-bit RGB values.

        Args:
            seg_id: Segment ID
            r1, g1, b1: Primary color (0-255)
            p1: Primary pattern (0-16)
            r2, g2, b2: Secondary color (0-255)
            p2: Secondary pattern (0-16)

        Returns:
            Result code (1 if success, 0 if failed)
        """
        # Encode colors as 32-bit values
        v1 = np.uint32(0)
        v2 = np.uint32(0)

        ir1 = np.uint32(r1)
        ig1 = np.uint32(g1)
        ib1 = np.uint32(b1)
        ip1 = np.uint32(p1)

        ir2 = np.uint32(r2)
        ig2 = np.uint32(g2)
        ib2 = np.uint32(b2)
        ip2 = np.uint32(p2)

        v1 = v1 + (ip1 & 255) + ((ib1 & 255) << 8) + ((ig1 & 255) << 16) + ((ir1 & 255) << 24)
        v2 = v2 + (ip2 & 255) + ((ib2 & 255) << 8) + ((ig2 & 255) << 16) + ((ir2 & 255) << 24)

        message = self._bytes_from_uint32(np.array([seg_id, v1, v2]))
        self._send_message(self.SETSEGMENTCOLOR, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_segment_color32(self, seg_id: int, col1: int, col2: int) -> int:
        """
        Set segment color using 32-bit color values.

        Args:
            seg_id: Segment ID
            col1: Primary color (32-bit)
            col2: Secondary color (32-bit)

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(np.array([seg_id, col1, col2]))
        self._send_message(self.SETSEGMENTCOLOR, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_anchor_point(self, seg_id: int, x: int, y: int, z: int) -> int:
        """
        Set segment anchor point.

        Args:
            seg_id: Segment ID
            x, y, z: Anchor coordinates

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(np.array([seg_id, x, y, z]))
        self._send_message(self.SETANCHORPOINT, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_segment_bbox(self, seg_id: int, x1: int, y1: int, z1: int,
                        x2: int, y2: int, z2: int) -> int:
        """
        Set segment bounding box.

        Args:
            seg_id: Segment ID
            x1, y1, z1: Min coordinates
            x2, y2, z2: Max coordinates

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(np.array([seg_id, x1, y1, z1, x2, y2, z2]))
        self._send_message(self.SETSEGMENTBBOX, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_selected_segment_nr(self) -> Tuple[int, int]:
        """
        Get currently selected segment number.

        Returns:
            Tuple: (segment_nr, result_code)
        """
        self._send_message(self.GETSELECTEDSEGMENTNR, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 1:
                seg_nr = int(self.in_uint_data[0])
                return seg_nr, res
            else:
                self.last_error = 2
                return 0, 0
        else:
            return 0, res

    def set_selected_segment_nr(self, seg_id: int) -> int:
        """
        Set currently selected segment.

        Args:
            seg_id: Segment ID to select

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(seg_id)
        self._send_message(self.SETSELECTEDSEGMENTNR, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_first_segment_nr(self) -> Tuple[int, int]:
        """
        Get first segment number.

        Returns:
            Tuple: (segment_nr, result_code)
        """
        self._send_message(self.GETFIRSTSEGMENTNR, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 1:
                seg_nr = int(self.in_uint_data[0])
                return seg_nr, res
            else:
                self.last_error = 2
                return 0, 0
        else:
            return 0, res

    def add_segment(self, parent_id: int, name: str) -> Tuple[int, int]:
        """
        Add new segment.

        Args:
            parent_id: Parent segment ID
            name: Segment name

        Returns:
            Tuple: (new_segment_id, result_code)
        """
        message = self._bytes_from_uint32(parent_id) + self._bytes_from_text(name)
        self._send_message(self.ADDSEGMENT, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 1:
                new_id = int(self.in_uint_data[0])
                return new_id, res
            else:
                self.last_error = 2
                return 0, 0
        else:
            return 0, res

    def move_segment(self, seg_id: int, new_parent_id: int) -> int:
        """
        Move segment to new parent.

        Args:
            seg_id: Segment ID to move
            new_parent_id: New parent segment ID

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(np.array([seg_id, new_parent_id]))
        self._send_message(self.MOVESEGMENT, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_seg_translation(self, seg_id: int, dx: int, dy: int, dz: int) -> int:
        """
        Set segment translation.

        Args:
            seg_id: Segment ID
            dx, dy, dz: Translation offset

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(np.array([seg_id, dx, dy, dz]))
        self._send_message(self.SETSEGTRANSLATION, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    # ============================================================================
    # LAYER MANAGEMENT
    # ============================================================================

    def get_layer_info(self, layer_nr: int) -> Tuple[Dict, int]:
        """
        Get layer information.

        Args:
            layer_nr: Layer number

        Returns:
            Tuple: (layer_info_dict, result_code)
        """
        message = self._bytes_from_uint32(layer_nr)
        self._send_message(self.GETLAYERINFO, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_int_data) == 8 and len(self.in_uint_data) == 7 and len(self.in_double_data) == 3:
                layerinfo = {
                    'type': int(self.in_int_data[0]),
                    'editable': int(self.in_int_data[1]),
                    'visible': int(self.in_int_data[2]),
                    'brightness': int(self.in_int_data[3]),
                    'contrast': int(self.in_int_data[4]),
                    'opacitylevel': float(self.in_double_data[0]),
                    'brightnesslevel': float(self.in_double_data[1]),
                    'contrastlevel': float(self.in_double_data[2]),
                    'blendmode': int(self.in_int_data[5]),
                    'blendoradd': int(self.in_int_data[6]),
                    'tintcolor': int(self.in_uint_data[0]),
                    'name': self.in_char_data,
                    'redtargetcolor': int(self.in_uint_data[1]),
                    'greentargetcolor': int(self.in_uint_data[2]),
                    'bluetargetcolor': int(self.in_uint_data[3]),
                    'bytesperpixel': int(self.in_uint_data[4]),
                    'ischanged': int(self.in_int_data[7]),
                    'inverted': int(self.in_uint_data[5]),
                    'solomode': int(self.in_uint_data[6]),
                }
                return layerinfo, res
            else:
                self.last_error = 2
                return {}, 0
        else:
            return {}, res

    def set_layer_info(self, layer_nr: int, layerinfo: Dict) -> int:
        """
        Set layer information.

        Args:
            layer_nr: Layer number
            layerinfo: Dictionary with layer properties

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(layer_nr)
        message += self._bytes_from_int32(layerinfo.get('type', 0))
        message += self._bytes_from_int32(layerinfo.get('editable', 0))
        message += self._bytes_from_int32(layerinfo.get('visible', 0))
        message += self._bytes_from_int32(layerinfo.get('brightness', 0))
        message += self._bytes_from_int32(layerinfo.get('contrast', 0))
        message += self._bytes_from_double(layerinfo.get('opacitylevel', 1.0))
        message += self._bytes_from_double(layerinfo.get('brightnesslevel', 0.0))
        message += self._bytes_from_double(layerinfo.get('contrastlevel', 1.0))
        message += self._bytes_from_int32(layerinfo.get('blendmode', 0))
        message += self._bytes_from_int32(layerinfo.get('blendoradd', 0))
        message += self._bytes_from_uint32(layerinfo.get('tintcolor', 0))
        message += self._bytes_from_text(layerinfo.get('name', ''))
        message += self._bytes_from_uint32(layerinfo.get('redtargetcolor', 0))
        message += self._bytes_from_uint32(layerinfo.get('greentargetcolor', 0))
        message += self._bytes_from_uint32(layerinfo.get('bluetargetcolor', 0))
        message += self._bytes_from_uint32(layerinfo.get('inverted', 0))
        message += self._bytes_from_uint32(layerinfo.get('solomode', 0))

        self._send_message(self.SETLAYERINFO, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_selected_layer_nr(self) -> Tuple[int, int]:
        """
        Get currently selected layer number.

        Returns:
            Tuple: (layer_nr, result_code)
        """
        self._send_message(self.GETSELECTEDLAYERNR, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 1:
                layer_nr = int(self.in_uint_data[0])
                return layer_nr, res
            else:
                self.last_error = 2
                return 0, 0
        else:
            return 0, res

    def get_selected_layer_nr2(self) -> Tuple[int, int, int]:
        """
        Get currently selected layer number and type.

        Returns:
            Tuple: (layer_nr, layer_type, result_code)
        """
        self._send_message(self.GETSELECTEDLAYERNR, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 2:
                layer_nr = int(self.in_uint_data[0])
                layer_type = int(self.in_uint_data[1])
                return layer_nr, layer_type, res
            elif len(self.in_uint_data) >= 1:
                layer_nr = int(self.in_uint_data[0])
                return layer_nr, 0, res
            else:
                self.last_error = 2
                return 0, 0, 0
        else:
            return 0, 0, res

    def set_selected_layer_nr(self, layer_nr: int) -> int:
        """
        Set currently selected layer.

        Args:
            layer_nr: Layer number to select

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(layer_nr)
        self._send_message(self.SETSELECTEDLAYERNR, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def add_new_layer(self, layer_type: int, name: str) -> Tuple[int, int]:
        """
        Add new layer.

        Args:
            layer_type: Layer type (0=seg, 1=EM, etc.)
            name: Layer name

        Returns:
            Tuple: (new_layer_nr, result_code)
        """
        message = self._bytes_from_uint32(layer_type) + self._bytes_from_text(name)
        self._send_message(self.ADDNEWLAYER, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_uint_data) >= 1:
                new_layer_nr = int(self.in_uint_data[0])
                return new_layer_nr, res
            else:
                self.last_error = 2
                return 0, 0
        else:
            return 0, res

    def load_layer(self, layer_nr: int, filename: str) -> int:
        """
        Load layer from file.

        Args:
            layer_nr: Layer number
            filename: File path

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(layer_nr) + self._bytes_from_text(filename)
        self._send_message(self.LOADLAYER, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def save_layer(self, layer_nr: int, filename: str) -> int:
        """
        Save layer to file.

        Args:
            layer_nr: Layer number
            filename: File path

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(layer_nr) + self._bytes_from_text(filename)
        self._send_message(self.SAVELAYER, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def remove_layer(self, layer_nr: int) -> int:
        """
        Remove layer.

        Args:
            layer_nr: Layer number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(layer_nr)
        self._send_message(self.REMOVELAYER, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def move_layer(self, layer_nr: int, new_position: int) -> int:
        """
        Move layer to new position.

        Args:
            layer_nr: Layer number
            new_position: New position index

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(np.array([layer_nr, new_position]))
        self._send_message(self.MOVELAYER, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_mipmap_scale_factors(self, layer_nr: int) -> Tuple[np.ndarray, int]:
        """
        Get MIP map scale factors for layer.

        Args:
            layer_nr: Layer number

        Returns:
            Tuple: (mip_scale_matrix, result_code)
                - mip_scale_matrix: Nx3 array where each row is [xscale, yscale, zscale] for that mip level
                  Row 0 = mip level 1, Row 1 = mip level 2, etc.
                  (mip level 0 is always [1,1,1] and is not included, matching MATLAB behavior)
                - result_code: 1 if success, 0 if failed
        """
        message = self._bytes_from_uint32(layer_nr)
        self._send_message(self.GETMIPMAPSCALEFACTORS, message)
        
        # Read with helper for potentially large data
        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parse_header_len + 12

        self.indata = bytearray(expected_length)
        self.indata[:len(indata1)] = indata1
        writepos = len(indata1)

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        self.indata = bytes(self.indata)
        
        if self.in_res == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            # Parse the raw data like MATLAB does
            uid = np.frombuffer(self.indata[16:], dtype=np.uint32)
            
            num_mip_levels = int(uid[0])
            if num_mip_levels == 0:
                return np.array([]), res
            
            # Build Nx3 matrix
            mip_scale_matrix = np.zeros((num_mip_levels, 3), dtype=np.float64)
            sp = 1  # Start after count
            
            for i in range(num_mip_levels):
                mip_scale_matrix[i, 0] = uid[sp]      # X scale
                mip_scale_matrix[i, 1] = uid[sp + 1]  # Y scale
                mip_scale_matrix[i, 2] = uid[sp + 2]  # Z scale
                sp += 3
            
            # MATLAB: mipscalematrix(1,:)=[]; - remove first row (mip level 0)
            if num_mip_levels > 1:
                mip_scale_matrix = mip_scale_matrix[1:, :]
            else:
                mip_scale_matrix = np.array([])
            
            return mip_scale_matrix, res
        else:
            return np.array([]), res

    # ============================================================================
    # VIEW CONTROL
    # ============================================================================

    def get_view_coordinates(self) -> Tuple[int, int, int, int]:
        """
        Get current view coordinates.

        Returns:
            Tuple: (x, y, z, result_code) - coordinates at mip0
        """
        self._send_message(self.GETVIEWCOORDINATES, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_int_data) >= 3:
                x = int(self.in_int_data[0])
                y = int(self.in_int_data[1])
                z = int(self.in_int_data[2])
                return x, y, z, res
            else:
                self.last_error = 2
                return 0, 0, 0, 0
        else:
            return 0, 0, 0, res

    def set_view_coordinates(self, x: int, y: int, z: int) -> int:
        """
        Set view coordinates.

        Args:
            x, y, z: Coordinates at mip0

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32(np.array([x, y, z]))
        self._send_message(self.SETVIEWCOORDINATES, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_view_zoom(self) -> Tuple[int, int]:
        """
        Get current zoom level.

        Returns:
            Tuple: (zoom, result_code)
        """
        self._send_message(self.GETVIEWZOOM, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if len(self.in_int_data) >= 1:
                zoom = int(self.in_int_data[0])
                return zoom, res
            else:
                self.last_error = 2
                return 0, 0
        else:
            return 0, res

    def set_view_zoom(self, zoom: int) -> int:
        """
        Set zoom level.

        Args:
            zoom: Zoom level

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_int32(zoom)
        self._send_message(self.SETVIEWZOOM, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    # ============================================================================
    # IMAGE RETRIEVAL - SEGMENTATION
    # ============================================================================

    def get_seg_image_raw(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                          minz: int, maxz: int, flipflag: int = 0, immediateflag: int = 0,
                          requestloadflag: int = 0) -> Tuple[np.ndarray, int]:
        """
        Get raw segmentation image data.

        Args:
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            flipflag: If 1, transpose XY dimensions
            immediateflag: If 1, use immediate mode
            requestloadflag: Request load flag for immediate mode

        Returns:
            Tuple of (segmentation image as uint16 array, result code)
        """
        if immediateflag == 0:
            message = self._bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz])
            self._send_message(self.GETSEGIMAGERAW, message)
        else:
            message = self._bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz, requestloadflag])
            self._send_message(self.GETSEGIMAGERAWIMMEDIATE, message)

        # Read data with helper for large transfers
        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        # Allocate buffer and read remaining data
        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            segimage = np.frombuffer(self.indata[16:].tobytes(), dtype=np.uint16)
            if flipflag == 1:
                xs, ys, zs = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
                segimage = segimage.reshape((xs, ys, zs), order='F')
                segimage = np.transpose(segimage, (1, 0, 2))
                segimage = segimage.flatten(order='F')
        else:
            segimage = np.array([], dtype=np.uint16)

        return segimage, res

    def get_seg_image_rle(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                          minz: int, maxz: int, surfonlyflag: int = 0, immediateflag: int = 0,
                          requestloadflag: int = 0) -> Tuple[np.ndarray, int]:
        """
        Get RLE-encoded segmentation image data.

        Args:
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            surfonlyflag: If 1, get surface only
            immediateflag: If 1, use immediate mode
            requestloadflag: Request load flag for immediate mode

        Returns:
            Tuple of (RLE data as uint16 array, result code)
        """
        if immediateflag == 0:
            if surfonlyflag == 0:
                message = self._bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz])
                self._send_message(self.GETSEGIMAGERLE, message)
            else:
                message = self._bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz])
                self._send_message(self.GETSEGIMAGESURFRLE, message)
        else:
            message = self._bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz, requestloadflag])
            self._send_message(self.GETSEGIMAGERLEIMMEDIATE, message)

        # Read data with helper for large transfers
        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            segimage_rle = np.frombuffer(self.indata[16:].tobytes(), dtype=np.uint16)
        else:
            segimage_rle = np.array([], dtype=np.uint16)

        return segimage_rle, res

    def get_seg_image_rle_decoded(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                                   minz: int, maxz: int, surfonlyflag: int = 0, flipflag: int = 0,
                                   immediateflag: int = 0, requestloadflag: int = 0) -> Tuple[np.ndarray, int]:
        """
        Get decoded segmentation image from RLE data.

        Args:
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            surfonlyflag: If 1, get surface only
            flipflag: If 1, transpose XY dimensions
            immediateflag: If 1, use immediate mode
            requestloadflag: Request load flag for immediate mode

        Returns:
            Tuple of (decoded segmentation image as uint16 array, result code)
        """
        segimage_rle, res = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                                     surfonlyflag, immediateflag, requestloadflag)

        if res == 0:
            return np.array([], dtype=np.uint16), res

        # Decode RLE
        xs, ys, zs = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
        # CRITICAL: Use Fortran order to match MATLAB's column-major array storage
        segimage = np.zeros((xs, ys, zs), dtype=np.uint16, order='F')
        dp = 0

        for sp in range(0, len(segimage_rle), 2):
            val = segimage_rle[sp]
            num = segimage_rle[sp + 1]
            segimage.flat[dp:dp + num] = val
            dp += num

        if flipflag == 1:
            segimage = np.transpose(segimage, (1, 0, 2))

        return segimage, res

    def get_rle_count_unique(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                             minz: int, maxz: int, surfonlyflag: int = 0, immediateflag: int = 0,
                             requestloadflag: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Get unique segment values and counts from RLE data.

        Args:
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            surfonlyflag: If 1, get surface only
            immediateflag: If 1, use immediate mode
            requestloadflag: Request load flag for immediate mode

        Returns:
            Tuple of (unique values array, counts array, result code)
        """
        segimage_rle, res = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                                     surfonlyflag, immediateflag, requestloadflag)

        if res == 0:
            return np.array([]), np.array([]), res

        # Count unique values
        maxsegval = np.max(segimage_rle[::2])
        na = np.zeros(maxsegval + 1, dtype=np.int64)

        for sp in range(0, len(segimage_rle), 2):
            val = segimage_rle[sp]
            num = segimage_rle[sp + 1]
            na[val] += num

        values = np.where(na > 0)[0]
        numbers = na[values]

        return values, numbers, res

    def get_seg_image_rle_decoded_count_unique(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                                                 minz: int, maxz: int, surfonlyflag: int = 0, flipflag: int = 0,
                                                 immediateflag: int = 0, requestloadflag: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Get decoded segmentation image and unique value counts.

        Args:
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            surfonlyflag: If 1, get surface only
            flipflag: If 1, transpose XY dimensions
            immediateflag: If 1, use immediate mode
            requestloadflag: Request load flag for immediate mode

        Returns:
            Tuple of (decoded image, unique values, counts, result code)
        """
        segimage_rle, res = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                                     surfonlyflag, immediateflag, requestloadflag)

        if res == 0:
            return np.array([]), np.array([]), np.array([]), res

        # Count unique and decode simultaneously
        maxsegval = np.max(segimage_rle[::2])
        na = np.zeros(maxsegval + 1, dtype=np.int64)

        xs, ys, zs = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
        # CRITICAL: Use Fortran order to match MATLAB's column-major array storage
        segimage = np.zeros((xs, ys, zs), dtype=np.uint16, order='F')
        dp = 0

        for sp in range(0, len(segimage_rle), 2):
            val = segimage_rle[sp]
            num = segimage_rle[sp + 1]
            segimage.flat[dp:dp + num] = val
            na[val] += num
            dp += num

        values = np.where(na > 0)[0]
        numbers = na[values]

        if flipflag == 1:
            segimage = np.transpose(segimage, (1, 0, 2))

        return segimage, values, numbers, res

    def get_seg_image_rle_decoded_bboxes(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                                          minz: int, maxz: int, surfonlyflag: int = 0, flipflag: int = 0,
                                          immediateflag: int = 0, requestloadflag: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Get decoded segmentation image with unique values, counts and bounding boxes.

        Args:
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            surfonlyflag: If 1, get surface only
            flipflag: If 1, transpose XY dimensions
            immediateflag: If 1, use immediate mode
            requestloadflag: Request load flag for immediate mode

        Returns:
            Tuple of (decoded image, unique values, counts, bboxes, result code)
        """
        segimage_rle, res = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                                     surfonlyflag, immediateflag, requestloadflag)

        if res == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), res

        maxsegval = np.max(segimage_rle[::2])
        na = np.zeros(maxsegval + 1, dtype=np.int32)
        bboxes = np.full((maxsegval + 1, 6), -1, dtype=np.int32)

        xs, ys, zs = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
        
        # CRITICAL FIX: Use Fortran order to match MATLAB's column-major array storage
        # MATLAB linearizes arrays in column-major (Fortran) order: X varies fastest, then Y, then Z
        # NumPy default is row-major (C) order: Z varies fastest, then Y, then X
        # We create the array in Fortran order so .flat iterates X->Y->Z like MATLAB
        segimage = np.zeros((xs, ys, zs), dtype=np.uint16, order='F')
        
        dp = 1
        x1, y1, z1 = 1, 1, 1

        for sp in range(0, len(segimage_rle), 2):
            val = segimage_rle[sp]
            num = int(segimage_rle[sp + 1])
            dp2 = dp + num - 1

            # With Fortran-ordered array, .flat now iterates in the same order as MATLAB
            segimage.flat[dp-1:dp2] = val
            na[val] += num

            # Bounding box computation (unchanged - matches MATLAB logic)
            if (x1 + num - 1) <= xs:
                xmin, xmax = x1, x1 + num - 1
                ymin, ymax = y1, y1
                zmin, zmax = z1, z1
                x1 = x1 + num
            else:
                z1 = (dp - 1) // (xs * ys) + 1
                r = dp - ((z1 - 1) * xs * ys)
                y1 = (r - 1) // xs + 1
                x1 = r - ((y1 - 1) * xs)

                z2 = (dp2 - 1) // (xs * ys) + 1
                r = dp2 - ((z2 - 1) * xs * ys)
                y2 = (r - 1) // xs + 1
                x2 = r - ((y2 - 1) * xs)

                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                zmin, zmax = min(z1, z2), max(z1, z2)

                if zmax > zmin:
                    xmin, xmax = 1, xs
                    ymin, ymax = 1, ys
                if ymax > ymin:
                    xmin, xmax = 1, xs

                x1, y1, z1 = x2 + 1, y2, z2

            if bboxes[val, 0] == -1:
                bboxes[val, :] = [xmin, ymin, zmin, xmax, ymax, zmax]
            else:
                bboxes[val, 0] = min(bboxes[val, 0], xmin)
                bboxes[val, 1] = min(bboxes[val, 1], ymin)
                bboxes[val, 2] = min(bboxes[val, 2], zmin)
                bboxes[val, 3] = max(bboxes[val, 3], xmax)
                bboxes[val, 4] = max(bboxes[val, 4], ymax)
                bboxes[val, 5] = max(bboxes[val, 5], zmax)

            dp = dp2 + 1

        values = np.where(na > 0)[0]
        numbers = na[values]
        bboxes = bboxes[values, :]

        if flipflag == 1:
            segimage = np.transpose(segimage, (1, 0, 2))

        return segimage, values, numbers, bboxes, res

    # ============================================================================
    # IMAGE RETRIEVAL - EM & SCREENSHOTS
    # ============================================================================

    def get_em_image_raw(self, layer_nr: int, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                         minz: int, maxz: int, immediateflag: int = 0, requestloadflag: int = 0) -> Tuple[np.ndarray, int]:
        """
        Get raw EM image data.

        Args:
            layer_nr: Layer number
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            immediateflag: If 1, use immediate mode
            requestloadflag: Request load flag for immediate mode

        Returns:
            Tuple of (EM image as uint8 array, result code)
        """
        if immediateflag == 0:
            message = self._bytes_from_uint32([layer_nr, miplevel, minx, maxx, miny, maxy, minz, maxz])
            self._send_message(self.GETEMIMAGERAW, message)
        else:
            message = self._bytes_from_uint32([layer_nr, miplevel, minx, maxx, miny, maxy, minz, maxz, requestloadflag])
            self._send_message(self.GETEMIMAGERAWIMMEDIATE, message)

        # Read data with helper for large transfers
        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            emimage = np.frombuffer(self.indata[16:].tobytes(), dtype=np.uint8)
        else:
            emimage = np.array([], dtype=np.uint8)

        return emimage, res

    def get_em_image(self, layer_nr: int, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                     minz: int, maxz: int, flipflag: int = 0, immediateflag: int = 0,
                     requestloadflag: int = 0) -> Tuple[np.ndarray, int]:
        """
        Get EM image data with proper reshaping.

        Args:
            layer_nr: Layer number
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            flipflag: If 1, transpose dimensions
            immediateflag: If 1, use immediate mode
            requestloadflag: Request load flag for immediate mode

        Returns:
            Tuple of (EM image array, result code)
        """
        emimageraw, res = self.get_em_image_raw(layer_nr, miplevel, minx, maxx, miny, maxy, minz, maxz,
                                                 immediateflag, requestloadflag)

        if res == 0:
            return emimageraw, res

        xs, ys, zs = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
        bytesppx = len(emimageraw) // (xs * ys * zs)

        if bytesppx == 1:
            # One byte per pixel
            if minz == maxz:
                emimage = emimageraw.reshape((xs, ys), order='F').T
            else:
                emimage = emimageraw.reshape((xs, ys, zs), order='F').transpose((1, 0, 2))
        elif bytesppx == 3:
            # Three bytes per pixel (RGB)
            if minz == maxz:
                emimage = emimageraw.reshape((3, xs, ys), order='F').transpose((2, 1, 0))[:, :, ::-1]
            else:
                emimage = emimageraw.reshape((3, xs, ys, zs), order='F').transpose((2, 1, 3, 0))[:, :, :, ::-1]
        elif bytesppx == 4:
            # Four bytes per pixel
            emimageraw = np.frombuffer(emimageraw.tobytes(), dtype=np.uint32)
            if minz == maxz:
                emimage = emimageraw.reshape((xs, ys), order='F').T
            else:
                emimage = emimageraw.reshape((xs, ys, zs), order='F').transpose((1, 0, 2))
        elif bytesppx == 8:
            # Eight bytes per pixel
            emimageraw = np.frombuffer(emimageraw.tobytes(), dtype=np.uint64)
            if minz == maxz:
                emimage = emimageraw.reshape((xs, ys), order='F').T
            else:
                emimage = emimageraw.reshape((xs, ys, zs), order='F').transpose((1, 0, 2))
        else:
            emimage = emimageraw

        return emimage, res

    def get_screenshot_image_raw(self, vp: int, minx: int, maxx: int, miny: int, maxy: int) -> Tuple[np.ndarray, int]:
        """
        Get raw screenshot image data.

        Args:
            vp: Viewport number
            minx, maxx, miny, maxy: Screenshot region coordinates

        Returns:
            Tuple of (screenshot image as uint8 array, result code)
        """
        message = self._bytes_from_uint32([vp, minx, maxx, miny, maxy, 0, 0, 0])
        self._send_message(self.GETSCREENSHOTIMAGERAW, message)

        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            screenshot = np.frombuffer(self.indata[16:].tobytes(), dtype=np.uint8)
        else:
            screenshot = np.array([], dtype=np.uint8)

        return screenshot, res

    def get_screenshot_image_rle(self, vp: int, minx: int, maxx: int, miny: int, maxy: int) -> Tuple[np.ndarray, int]:
        """
        Get RLE-encoded screenshot image data.

        Args:
            vp: Viewport number
            minx, maxx, miny, maxy: Screenshot region coordinates

        Returns:
            Tuple of (screenshot image as uint8 array, result code)
        """
        message = self._bytes_from_uint32([vp, minx, maxx, miny, maxy, 0, 0, 0])
        self._send_message(self.GETSCREENSHOTIMAGERLE, message)

        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            rledata = np.frombuffer(self.indata[16:].tobytes(), dtype=np.uint8)

            expected_size = (maxx - minx + 1) * (maxy - miny + 1) * 3
            if len(rledata) == expected_size:
                screenshot = rledata
            else:
                # Decode RLE
                screenshot = np.zeros(expected_size, dtype=np.uint8)
                dp = 0

                for sp in range(0, len(rledata), 4):
                    val1 = rledata[sp]
                    val2 = rledata[sp + 1]
                    val3 = rledata[sp + 2]
                    num = rledata[sp + 3]
                    screenshot[dp::3][:num] = val1
                    screenshot[dp+1::3][:num] = val2
                    screenshot[dp+2::3][:num] = val3
                    dp += num * 3
        else:
            screenshot = np.array([], dtype=np.uint8)

        return screenshot, res

    def get_screenshot_image(self, vp: int, minx: int, maxx: int, miny: int, maxy: int,
                             flipflag: int = 0) -> Tuple[np.ndarray, int]:
        """
        Get screenshot image with proper reshaping.

        Args:
            vp: Viewport number
            minx, maxx, miny, maxy: Screenshot region coordinates
            flipflag: If 1, use RLE encoding

        Returns:
            Tuple of (screenshot image array, result code)
        """
        if flipflag:
            screenshotraw, res = self.get_screenshot_image_rle(vp, minx, maxx, miny, maxy)
        else:
            screenshotraw, res = self.get_screenshot_image_raw(vp, minx, maxx, miny, maxy)

        if res == 0:
            return screenshotraw, res

        xs, ys = maxx - minx + 1, maxy - miny + 1
        screenshot = screenshotraw.reshape((3, xs, ys), order='F').transpose((2, 1, 0))[:, :, ::-1]

        return screenshot, res

    def order_screenshot_image(self, vp: int, minx: int, maxx: int, miny: int, maxy: int) -> int:
        """
        Order screenshot image (send request without waiting for response).

        Args:
            vp: Viewport number
            minx, maxx, miny, maxy: Screenshot region coordinates

        Returns:
            Result code (always 1)
        """
        message = self._bytes_from_uint32([vp, minx, maxx, miny, maxy, 0, 0, 0])
        self._send_message(self.GETSCREENSHOTIMAGERAW, message)
        return 1

    def pickup_screenshot_image(self) -> Tuple[np.ndarray, int]:
        """
        Pickup screenshot image (read response from previous order).

        Returns:
            Tuple of (screenshot image array, result code)
        """
        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            screenshot = np.frombuffer(self.indata[16:].tobytes(), dtype=np.uint8)
        else:
            screenshot = np.array([], dtype=np.uint8)

        return screenshot, res

    # ============================================================================
    # IMAGE WRITING
    # ============================================================================

    def set_seg_image_raw(self, layer_nr: int, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                          minz: int, maxz: int, imagedata: np.ndarray, flipflag: int = 0) -> int:
        """
        Set raw segmentation image data.

        Args:
            layer_nr: Layer number
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            imagedata: Segmentation image as uint16 array
            flipflag: If 1, input is in transposed format

        Returns:
            Result code (1 if success, 0 if failed)
        """
        xs, ys, zs = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1

        if imagedata.shape != (ys, xs, zs) and imagedata.shape != (xs, ys, zs):
            self.lasterror = 13
            return 0

        mparams = self._bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz])

        if flipflag == 1:
            segimage = imagedata.transpose((1, 0, 2))
        else:
            segimage = imagedata.transpose((1, 0, 2))

        mdata = self._bytes_from_data(segimage.astype(np.uint16).flatten(order='F'))

        self._send_message(self.SETSEGIMAGERAW, mparams + mdata)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        return res

    def set_seg_image_rle(self, layer_nr: int, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                          minz: int, maxz: int, rledata: np.ndarray) -> int:
        """
        Set RLE-encoded segmentation image data.

        Args:
            layer_nr: Layer number
            miplevel: MIP level
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            rledata: RLE data as uint16 array

        Returns:
            Result code (1 if success, 0 if failed)
        """
        mparams = self._bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz])
        mdata = self._bytes_from_data(rledata.astype(np.uint16))

        self._send_message(self.SETSEGIMAGERLE, mparams + mdata)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        return res

    # ============================================================================
    # UI & MODE CONTROL
    # ============================================================================

    def get_current_ui_state(self) -> Tuple[Dict, int]:
        """
        Get current UI state.

        Returns:
            Tuple of (state dictionary, result code)
        """
        self._send_message(self.GETCURRENTUISTATE, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 5 and self.nrindoubles == 0 and self.nrinints == 9:
                state = {
                    'mousecoordsx': self.inintdata[0],
                    'mousecoordsy': self.inintdata[1],
                    'lastleftclickx': self.inintdata[2],
                    'lastleftclicky': self.inintdata[3],
                    'lastleftreleasex': self.inintdata[4],
                    'lastleftreleasey': self.inintdata[5],
                    'mousecoordsz': self.inintdata[6],
                    'clientwindowwidth': self.inintdata[7],
                    'clientwindowheight': self.inintdata[8],
                    'reservedflag': self.inuintdata[0] & 1,
                    'lbuttondown': (self.inuintdata[0] >> 1) & 1,
                    'rbuttondown': (self.inuintdata[0] >> 2) & 1,
                    'mbuttondown': (self.inuintdata[0] >> 3) & 1,
                    'ctrlpressed': (self.inuintdata[0] >> 4) & 1,
                    'shiftpressed': (self.inuintdata[0] >> 5) & 1,
                    'deletepressed': (self.inuintdata[0] >> 6) & 1,
                    'spacepressed': (self.inuintdata[0] >> 7) & 1,
                    'spacewaspressed': (self.inuintdata[0] >> 8) & 1,
                    'uimode': self.inuintdata[1],
                    'hoversegmentnr': self.inuintdata[2],
                    'miplevel': self.inuintdata[3],
                    'paintcursordiameter': self.inuintdata[4]
                }
            else:
                state = {}
                res = 0
                self.lasterror = 2
        else:
            state = {}

        return state, res

    def set_ui_mode(self, mode: int) -> int:
        """
        Set UI mode.

        Args:
            mode: UI mode value

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([mode])
        self._send_message(self.SETUIMODE, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def show_window(self, window_id: int) -> int:
        """
        Show/hide window.

        Args:
            window_id: Window identifier

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_int32([window_id, 1, 0, -1, -1, 0, 0, 0])
        self._send_message(self.SHOWWINDOW, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_2d_view_orientation(self, orientation: int) -> int:
        """
        Set 2D view orientation.

        Args:
            orientation: Orientation value

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_int32([orientation, 0])
        self._send_message(self.SET2DVIEWORIENTATION, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_error_popups_enabled(self) -> Tuple[int, int]:
        """
        Get error popups enabled status.

        Returns:
            Tuple of (enabled flag, result code)
        """
        message = self._bytes_from_uint32([0])
        self._send_message(self.GETERRORPOPUPSENABLED, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                enabled = self.inuintdata[0]
            else:
                enabled = 0
                res = 0
                self.lasterror = 2
        else:
            enabled = 0

        return enabled, res

    def set_error_popups_enabled(self, code: int = 0, enabled: bool = True) -> int:
        """
        Set error popups enabled status.

        Args:
            code: Error code (default 0 for all errors)
            enabled: Enable flag (True/False or 1/0)

        Returns:
            Result code (1 if success, 0 if failed)
        """
        enabled_int = 1 if enabled else 0
        message = self._bytes_from_uint32([code, enabled_int])
        self._send_message(self.SETERRORPOPUPSENABLED, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_popups_enabled(self) -> Tuple[int, int]:
        """
        Get popups enabled status.

        Returns:
            Tuple of (enabled flag, result code)
        """
        message = self._bytes_from_uint32([0])
        self._send_message(self.GETPOPUPSENABLED, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                enabled = self.inuintdata[0]
            else:
                enabled = 0
                res = 0
                self.lasterror = 2
        else:
            enabled = 0

        return enabled, res

    def set_popups_enabled(self, enabled: int) -> int:
        """
        Set popups enabled status.

        Args:
            enabled: Enable flag (1 or 0)

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([0, enabled])
        self._send_message(self.SETPOPUPSENABLED, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    # ============================================================================
    # API LAYER CONTROL
    # ============================================================================

    def get_api_layers_enabled(self) -> Tuple[int, int]:
        """
        Get API layers enabled status.

        Returns:
            Tuple of (enabled flag, result code)
        """
        self._send_message(self.GETAPILAYERSENABLED, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                enabled = self.inuintdata[0]
            else:
                enabled = 0
                res = 0
                self.lasterror = 2
        else:
            enabled = 0

        return enabled, res

    def set_api_layers_enabled(self, enabled: int) -> int:
        """
        Set API layers enabled status.

        Args:
            enabled: Enable flag (1 or 0)

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([enabled])
        self._send_message(self.SETAPILAYERSENABLED, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_selected_api_layer_nr(self) -> Tuple[int, int, int, int, int, int]:
        """
        Get selected API layer numbers.

        Returns:
            Tuple of (layer_nr, em_layer_nr, anno_layer_nr, segment_layer_nr, tool_layer_nr, result code)
        """
        self._send_message(self.GETSELECTEDAPILAYERNR, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinints == 5:
                return (self.inintdata[0], self.inintdata[1], self.inintdata[2],
                        self.inintdata[3], self.inintdata[4], res)
            else:
                res = 0
                self.lasterror = 2

        return -1, -1, -1, -1, -1, res

    def set_selected_api_layer_nr(self, layer_nr: int) -> int:
        """
        Set selected API layer number.

        Args:
            layer_nr: Layer number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_int32([layer_nr])
        self._send_message(self.SETSELECTEDAPILAYERNR, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    # ============================================================================
    # ANNOTATION OBJECTS
    # ============================================================================

    def get_anno_layer_nr_of_objects(self, layer_nr: int) -> Tuple[int, int]:
        """
        Get number of annotation objects in layer.

        Args:
            layer_nr: Layer number

        Returns:
            Tuple of (number of objects, result code)
        """
        self._send_message(self.GETANNOLAYERNROFOBJECTS, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 2:
                nr_objects = self.inuintdata[0]
            else:
                nr_objects = 0
                res = 0
                self.lasterror = 2
        else:
            nr_objects = 0

        return nr_objects, res

    def get_anno_layer_object_data(self, layer_nr: int, object_nr: int) -> Tuple[Dict, int]:
        """
        Get annotation layer object data.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Tuple of (object data dictionary, result code)
        """
        self._send_message(self.GETANNOLAYEROBJECTDATA, b'')

        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            uid = np.frombuffer(self.indata[16:].tobytes(), dtype=np.uint32)
            obj_data = {}
            # Parse object data structure
            # Implementation similar to MATLAB version
        else:
            obj_data = {}

        return obj_data, res

    def get_anno_layer_object_names(self, layer_nr: int) -> Tuple[List[str], int]:
        """
        Get annotation layer object names.

        Args:
            layer_nr: Layer number

        Returns:
            Tuple of (list of names, result code)
        """
        self._send_message(self.GETANNOLAYEROBJECTNAMES, b'')

        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            nrnames = np.frombuffer(self.indata[16:20].tobytes(), dtype=np.uint32)[0]
            names = []
            sp = 20

            for i in range(nrnames):
                sq = sp
                while self.indata[sq] != 0:
                    sq += 1
                name = self.indata[sp:sq].tobytes().decode('utf-8')
                names.append(name)
                sp = sq + 1
        else:
            names = []

        return names, res

    def add_new_anno_object(self, layer_nr: int, object_type: int, name: str) -> Tuple[int, int]:
        """
        Add new annotation object.

        Args:
            layer_nr: Layer number
            object_type: Object type (0: folder, 1: skeleton)
            name: Object name

        Returns:
            Tuple of (new object ID, result code)
        """
        message = self._bytes_from_uint32([0, 1, object_type]) + self._bytes_from_text(name)
        self._send_message(self.ADDNEWANNOOBJECT, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                obj_id = self.inuintdata[0]
            else:
                obj_id = 0
                res = 0
                self.lasterror = 2
        else:
            obj_id = 0

        return obj_id, res

    def move_anno_object(self, layer_nr: int, source_nr: int, target_nr: int) -> int:
        """
        Move annotation object.

        Args:
            layer_nr: Layer number
            source_nr: Source object number
            target_nr: Target position number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([source_nr, target_nr, 0])
        self._send_message(self.MOVEANNOOBJECT, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def remove_anno_object(self, layer_nr: int, object_nr: int) -> int:
        """
        Remove annotation object.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([object_nr])
        self._send_message(self.REMOVEANNOOBJECT, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_selected_anno_object_nr(self, object_nr: int) -> int:
        """
        Set selected annotation object number.

        Args:
            object_nr: Object number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([object_nr])
        self._send_message(self.SETSELECTEDANNOOBJECTNR, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_selected_anno_object_nr(self) -> Tuple[int, int]:
        """
        Get selected annotation object number.

        Returns:
            Tuple of (object number, result code)
        """
        self._send_message(self.GETSELECTEDANNOOBJECTNR, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                obj_nr = self.inuintdata[0]
            else:
                obj_nr = -1
                res = 0
                self.lasterror = 2
        else:
            obj_nr = -1

        return obj_nr, res

    def get_anno_object(self, layer_nr: int, object_nr: int) -> Tuple[Dict, int]:
        """
        Get annotation object data.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Tuple of (object data dict, result code)
        """
        message = self._bytes_from_uint32([object_nr])
        self._send_message(self.GETANNOOBJECT, message)

        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            # Decode struct from binary data
            aodata = {}
            # Implementation would decode the binary struct data
        else:
            aodata = {}

        return aodata, res

    def set_anno_object(self, layer_nr: int, object_nr: int, obj_data: Dict) -> int:
        """
        Set annotation object data.

        Args:
            layer_nr: Layer number
            object_nr: Object number
            obj_data: Object data dictionary

        Returns:
            Result code (1 if success, 0 if failed)
        """
        # Encode struct to binary
        encoded_data = b''  # Would encode obj_data to binary format

        message = self._bytes_from_uint32([object_nr]) + encoded_data
        self._send_message(self.SETANNOOBJECT, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                obj_id = self.inuintdata[0]
            else:
                res = 0
                self.lasterror = 2

        return res

    def add_anno_object(self, layer_nr: int, obj_data: Dict) -> Tuple[int, int]:
        """
        Add annotation object.

        Args:
            layer_nr: Layer number
            obj_data: Object data dictionary

        Returns:
            Tuple of (new object ID, result code)
        """
        # Encode struct to binary
        encoded_data = b''  # Would encode obj_data to binary format

        message = self._bytes_from_uint32([0, 1]) + encoded_data
        self._send_message(self.ADDANNOOBJECT, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                obj_id = self.inuintdata[0]
            else:
                obj_id = 0
                res = 0
                self.lasterror = 2
        else:
            obj_id = 0

        return obj_id, res

    # ============================================================================
    # ANNOTATION NODES
    # ============================================================================

    def get_ao_node_data(self, layer_nr: int, object_nr: int) -> Tuple[np.ndarray, int]:
        """
        Get annotation object node data.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Tuple of (node data array, result code)
        """
        self._send_message(self.GETAONODEDATA, b'')

        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            uid = np.frombuffer(self.indata[16:].tobytes(), dtype=np.uint32)
            nr_nodes = uid[0]
            aonodedata = np.zeros((nr_nodes, 14), dtype=np.float64)
            # Parse node data
            # Implementation similar to MATLAB version
        else:
            aonodedata = np.array([])

        return aonodedata, res

    def get_ao_node_labels(self, layer_nr: int, object_nr: int) -> Tuple[List[str], int]:
        """
        Get annotation object node labels.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Tuple of (list of labels, result code)
        """
        self._send_message(self.GETAONODELABELS, b'')

        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            nroflabels = np.frombuffer(self.indata[16:20].tobytes(), dtype=np.uint32)[0]
            labels = []
            # Parse labels similar to get_anno_layer_object_names
        else:
            labels = []

        return labels, res

    def get_ao_node_params(self, layer_nr: int, object_nr: int, node_dfs_nr: int) -> Tuple[Dict, int]:
        """
        Get annotation object node parameters.

        Args:
            layer_nr: Layer number
            object_nr: Object number
            node_dfs_nr: Node DFS number

        Returns:
            Tuple of (node parameters dict, result code)
        """
        message = self._bytes_from_uint32([object_nr, node_dfs_nr])
        self._send_message(self.GETAONODEPARAMS, message)

        indata1 = self._read_data_block()
        self._parse_header(indata1)
        expected_length = self.parseheaderlen + 12

        self.indata = np.zeros(expected_length, dtype=np.int8)
        writepos = len(indata1)
        self.indata[:len(indata1)] = indata1

        while writepos < expected_length:
            indata2 = self._read_data_block()
            self.indata[writepos:writepos+len(indata2)] = indata2
            writepos += len(indata2)

        if self.inres == 0:
            self._parse(self.indata)

        res = self._process_error()

        if res == 1:
            nodedata = {}
            # Decode node parameters
        else:
            nodedata = {}

        return nodedata, res

    def set_ao_node_params(self, layer_nr: int, object_nr: int, node_dfs_nr: int, params: Dict) -> int:
        """
        Set annotation object node parameters.

        Args:
            layer_nr: Layer number
            object_nr: Object number
            node_dfs_nr: Node DFS number
            params: Node parameters dictionary

        Returns:
            Result code (1 if success, 0 if failed)
        """
        encoded_data = b''  # Encode params to binary

        message = self._bytes_from_uint32([object_nr, node_dfs_nr]) + encoded_data
        self._send_message(self.SETAONODEPARAMS, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_selected_ao_node_by_dfsnr(self, layer_nr: int, object_nr: int, dfs_nr: int) -> int:
        """
        Set selected annotation object node by DFS number.

        Args:
            layer_nr: Layer number
            object_nr: Object number
            dfs_nr: DFS number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([dfs_nr])
        self._send_message(self.SETSELECTEDAONODEBYDFSNR, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_selected_ao_node_by_coords(self, layer_nr: int, object_nr: int, x: int, y: int, z: int) -> int:
        """
        Set selected annotation object node by coordinates.

        Args:
            layer_nr: Layer number
            object_nr: Object number
            x, y, z: Coordinates

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([x, y, z])
        self._send_message(self.SETSELECTEDAONODEBYCOORDS, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_selected_ao_node_nr(self, layer_nr: int, object_nr: int) -> Tuple[int, int]:
        """
        Get selected annotation object node number.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Tuple of (node number, result code)
        """
        self._send_message(self.GETSELECTEDAONODENR, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                node_nr = self.inuintdata[0]
            else:
                node_nr = -1
                res = 0
                self.lasterror = 2
        else:
            node_nr = -1

        return node_nr, res

    def add_ao_node(self, layer_nr: int, object_nr: int, parent_dfs_nr: int, x: int, y: int, z: int) -> Tuple[int, int]:
        """
        Add annotation object node.

        Args:
            layer_nr: Layer number
            object_nr: Object number
            parent_dfs_nr: Parent node DFS number
            x, y, z: Node coordinates

        Returns:
            Tuple of (new node number, result code)
        """
        message = self._bytes_from_uint32([x, y, z])
        self._send_message(self.ADDAONODE, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            node_nr = 0  # Would extract from response
        else:
            node_nr = -1

        return node_nr, res

    def move_selected_ao_node(self, layer_nr: int, object_nr: int, x: int, y: int, z: int) -> int:
        """
        Move selected annotation object node.

        Args:
            layer_nr: Layer number
            object_nr: Object number
            x, y, z: New coordinates

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([x, y, z])
        self._send_message(self.MOVESELECTEDAONODE, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def remove_selected_ao_node(self, layer_nr: int, object_nr: int) -> int:
        """
        Remove selected annotation object node.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        self._send_message(self.REMOVESELECTEDAONODE, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def swap_selected_ao_node_children(self, layer_nr: int, object_nr: int) -> int:
        """
        Swap children of selected annotation object node.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        self._send_message(self.SWAPSELECTEDAONODECHILDREN, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def make_selected_ao_node_root(self, layer_nr: int, object_nr: int) -> int:
        """
        Make selected annotation object node the root.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        self._send_message(self.MAKESELECTEDAONODEROOT, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_closest_ao_node_by_coords(self, layer_nr: int, object_nr: int, x: int, y: int, z: int) -> Tuple[int, int, float, int]:
        """
        Get closest annotation object node by coordinates.

        Args:
            layer_nr: Layer number
            object_nr: Object number
            x, y, z: Coordinates

        Returns:
            Tuple of (object ID, node DFS number, distance, result code)
        """
        message = self._bytes_from_uint32([x, y, z, 1000000])
        self._send_message(self.GETCLOSESTAONODEBYCOORDS, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinints == 2 and self.nrindoubles == 1:
                anno_id = self.inintdata[0]
                node_dfs_nr = self.inintdata[1]
                distance = self.indoubledata[0]
            else:
                anno_id, node_dfs_nr, distance = -1, -1, -1.0
        else:
            anno_id, node_dfs_nr, distance = -1, -1, -1.0

        return anno_id, node_dfs_nr, distance, res

    # ============================================================================
    # SKELETON OPERATIONS
    # ============================================================================

    def split_selected_skeleton(self, layer_nr: int, object_nr: int) -> int:
        """
        Split selected skeleton.

        Args:
            layer_nr: Layer number
            object_nr: Object number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([0]) + self._bytes_from_text("split")
        self._send_message(self.SPLITSELECTEDSKELETON, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 1:
                new_id = self.inuintdata[0]
            else:
                res = 0
                self.lasterror = 2

        return res

    def weld_skeletons(self, layer_nr: int, source_obj_nr: int, target_obj_nr: int) -> int:
        """
        Weld two skeletons together.

        Args:
            layer_nr: Layer number
            source_obj_nr: Source object number
            target_obj_nr: Target object number

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([source_obj_nr, 0, target_obj_nr, 0])
        self._send_message(self.WELDSKELETONS, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    # ============================================================================
    # FILL & DRAWING
    # ============================================================================

    def get_drawing_properties(self, layer_nr: int) -> Tuple[Dict, int]:
        """
        Get drawing properties.

        Args:
            layer_nr: Layer number

        Returns:
            Tuple of (properties dict, result code)
        """
        self._send_message(self.GETDRAWINGPROPERTIES, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 7 and self.nrinints == 1 and self.nrindoubles == 2:
                props = {
                    'paintcursordiameter': self.inuintdata[0],
                    'paintcursorlocked': self.inuintdata[1] & 1,
                    'autofill': (self.inuintdata[1] >> 1) & 1,
                    'zscrollenabled': (self.inuintdata[1] >> 2) & 1,
                    'overwritemode': self.inuintdata[2],
                    'mippaintrestriction': self.inintdata[0],
                    'paintdepth': self.inuintdata[3],
                    'useconditionalpainting': self.inuintdata[4] & 1,
                    'cp_contiguousonly': (self.inuintdata[4] >> 1) & 1,
                    'cp_method': self.inuintdata[5],
                    'cp_sourcelayernr': self.inuintdata[6],
                    'cp_lowvalue': self.indoubledata[0],
                    'cp_highvalue': self.indoubledata[1]
                }
            else:
                props = {}
                res = 0
                self.lasterror = 2
        else:
            props = {}

        return props, res

    def set_drawing_properties(self, layer_nr: int, props: Dict) -> int:
        """
        Set drawing properties.

        Args:
            layer_nr: Layer number
            props: Properties dictionary

        Returns:
            Result code (1 if success, 0 if failed)
        """
        xflags = 0
        msg = b''

        if 'paintcursorlocked' in props:
            msg += self._bytes_from_int32([props['paintcursorlocked']])
            xflags += 1
        if 'paintcursordiameter' in props:
            msg += self._bytes_from_uint32([props['paintcursordiameter']])
            xflags += 2
        if 'autofill' in props:
            msg += self._bytes_from_int32([props['autofill']])
            xflags += 4
        if 'zscrollenabled' in props:
            msg += self._bytes_from_int32([props['zscrollenabled']])
            xflags += 8
        if 'overwritemode' in props:
            msg += self._bytes_from_uint32([props['overwritemode']])
            xflags += 16
        if 'mippaintrestriction' in props:
            msg += self._bytes_from_int32([props['mippaintrestriction']])
            xflags += 32
        if 'paintdepth' in props:
            msg += self._bytes_from_uint32([props['paintdepth']])
            xflags += 64
        if 'useconditionalpainting' in props:
            msg += self._bytes_from_int32([props['useconditionalpainting']])
            xflags += 128
        if 'cp_contiguousonly' in props:
            msg += self._bytes_from_int32([props['cp_contiguousonly']])
            xflags += 256
        if 'cp_method' in props:
            msg += self._bytes_from_uint32([props['cp_method']])
            xflags += 512
        if 'cp_sourcelayernr' in props:
            msg += self._bytes_from_int32([props['cp_sourcelayernr']])
            xflags += 1024
        if 'cp_lowvalue' in props:
            msg += self._bytes_from_double([props['cp_lowvalue']])
            xflags += 2048
        if 'cp_highvalue' in props:
            msg += self._bytes_from_double([props['cp_highvalue']])
            xflags += 4096

        message = self._bytes_from_uint32([xflags]) + msg
        self._send_message(self.SETDRAWINGPROPERTIES, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def get_filling_properties(self, layer_nr: int) -> Tuple[Dict, int]:
        """
        Get filling properties.

        Args:
            layer_nr: Layer number

        Returns:
            Tuple of (properties dict, result code)
        """
        self._send_message(self.GETFILLINGPROPERTIES, b'')
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuints == 3 and self.nrinints == 2 and self.nrindoubles == 6:
                props = {
                    'overwritemode': self.inuintdata[0],
                    'mippaintrestriction': self.inintdata[0],
                    'donotfillsourcecolorzero': self.inuintdata[1] & 1,
                    'sourcelayersameastarget': (self.inuintdata[1] >> 1) & 1,
                    'sourcelayernr': self.inintdata[1],
                    'method': self.inuintdata[2],
                    'lowvalue_x': self.indoubledata[0],
                    'highvalue_x': self.indoubledata[1],
                    'lowvalue_y': self.indoubledata[2],
                    'highvalue_y': self.indoubledata[3],
                    'lowvalue_z': self.indoubledata[4],
                    'highvalue_z': self.indoubledata[5]
                }
            else:
                props = {}
                res = 0
                self.lasterror = 2
        else:
            props = {}

        return props, res

    def set_filling_properties(self, layer_nr: int, props: Dict) -> int:
        """
        Set filling properties.

        Args:
            layer_nr: Layer number
            props: Properties dictionary

        Returns:
            Result code (1 if success, 0 if failed)
        """
        xflags = 0
        msg = b''

        if 'overwritemode' in props:
            msg += self._bytes_from_int32([props['overwritemode']])
            xflags += 1
        if 'mippaintrestriction' in props:
            msg += self._bytes_from_int32([props['mippaintrestriction']])
            xflags += 2
        if 'donotfillsourcecolorzero' in props:
            msg += self._bytes_from_int32([props['donotfillsourcecolorzero']])
            xflags += 4
        if 'sourcelayersameastarget' in props:
            msg += self._bytes_from_int32([props['sourcelayersameastarget']])
            xflags += 8
        if 'sourcelayernr' in props:
            msg += self._bytes_from_int32([props['sourcelayernr']])
            xflags += 16
        if 'method' in props:
            msg += self._bytes_from_int32([props['method']])
            xflags += 32
        if 'lowvalue_x' in props:
            msg += self._bytes_from_double([props['lowvalue_x']])
            xflags += 64
        if 'highvalue_x' in props:
            msg += self._bytes_from_double([props['highvalue_x']])
            xflags += 128
        if 'lowvalue_y' in props:
            msg += self._bytes_from_double([props['lowvalue_y']])
            xflags += 256
        if 'highvalue_y' in props:
            msg += self._bytes_from_double([props['highvalue_y']])
            xflags += 512
        if 'lowvalue_z' in props:
            msg += self._bytes_from_double([props['lowvalue_z']])
            xflags += 1024
        if 'highvalue_z' in props:
            msg += self._bytes_from_double([props['highvalue_z']])
            xflags += 2048

        message = self._bytes_from_uint32([xflags]) + msg
        self._send_message(self.SETFILLINGPROPERTIES, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def execute_fill(self, source_layer_nr: int, target_layer_nr: int, x: int, y: int, z: int, mip: int) -> int:
        """
        Execute fill operation.

        Args:
            source_layer_nr: Source layer number
            target_layer_nr: Target layer number
            x, y, z: Fill start coordinates
            mip: MIP level

        Returns:
            Result code (1 if success, 0 if failed)
        """
        # This would need the appropriate message ID defined
        return 0

    def execute_limited_fill(self, source_layer_nr: int, target_layer_nr: int, x: int, y: int, z: int, mip: int) -> int:
        """
        Execute limited fill operation.

        Args:
            source_layer_nr: Source layer number
            target_layer_nr: Target layer number
            x, y, z: Fill start coordinates
            mip: MIP level

        Returns:
            Result code (1 if success, 0 if failed)
        """
        # This would need the appropriate message ID defined
        return 0

    # ============================================================================
    # UTILITY
    # ============================================================================

    def get_pixel_value_from_full_res_coords(self, layer_nr: int, x: int, y: int, z: int) -> Tuple[int, int]:
        """
        Get pixel value from full resolution coordinates.

        Args:
            layer_nr: Layer number
            x, y, z: Coordinates

        Returns:
            Tuple of (pixel value, result code)
        """
        message = self._bytes_from_uint32([layer_nr, 0, x, y, z])
        self._send_message(self.GETPIXELVALUE, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()

        if res == 1:
            if self.nrinuint64s == 1:
                pixel_value = self.inuint64data[0]
            else:
                pixel_value = 0
                res = 0
                self.lasterror = 2
        else:
            pixel_value = 0

        return pixel_value, res

    def refresh_layer_region(self, layer_nr: int, minx: int, maxx: int, miny: int, maxy: int, minz: int, maxz: int) -> int:
        """
        Refresh layer region.

        Args:
            layer_nr: Layer number
            minx, maxx, miny, maxy, minz, maxz: Region bounds

        Returns:
            Result code (1 if success, 0 if failed)
        """
        message = self._bytes_from_uint32([layer_nr, minx, maxx, miny, maxy, minz, maxz])
        self._send_message(self.REFRESHLAYERREGION, message)
        self._read_data_block()
        self._parse(self.indata)
        res = self._process_error()
        return res

    def set_tool_parameters(self, tool_layer_nr: int, tool_node_nr: int, params: Dict) -> int:
        """
        Set tool parameters.

        Args:
            tool_layer_nr: Tool layer number
            tool_node_nr: Tool node number
            params: Parameters dictionary

        Returns:
            Result code (1 if success, 0 if failed)
        """
        # Implementation depends on parameter encoding
        return 0

    def execute_canvas_paint_stroke(self, coords: np.ndarray) -> int:
        """
        Execute canvas paint stroke.

        Args:
            coords: Coordinates array

        Returns:
            Result code (1 if success, 0 if failed)
        """
        # Implementation depends on coordinate encoding
        return 0