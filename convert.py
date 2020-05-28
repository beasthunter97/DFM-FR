f = open('file_check.py', 'r')
out = open('cam_out.py', 'w')
in_ = open('cam_in.py', 'w')
data = f.read()
f.close()
data = data.replace('FileVideoStream(', 'CamVidStream(')
data = data.replace('FileVideoStream', 'WebcamVideoStream')
data = data.replace('import pickle', 'import pickle\nimport time')
string = '''# region DEF
class CamVidStream(WebcamVideoStream):
    def __init__(self, src, transform=None):
        WebcamVideoStream.__init__(self, src)
        self.transform = transform

    def update(self):
        while True:
            if self.stopped:
                return

            self.grabbed, self.frame = self.stream.read()
            self.frame = self.transform(self.frame)\n\n'''
data = data.replace('# region DEF', string)
data = data.replace('while video.more():', 'while True:')
string = '''        if frame is None:
            break
'''
data = data.replace(string, '')
data = data.replace('\n    writer = None', '')
string = '''
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args['output'], fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        if writer is not None:
            writer.write(frame)
'''
data = data.replace(string, '')
data = data.replace('    writer.release()\n', '')
string = """if args['input'] == 'cam':
        src = 'rtsp://192.168.200.80:555/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream' # noqa
        time.sleep(30)
    elif '192.168' in args['input']:
        src = 'rtsp://' + args['input'] + '/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream' # noqa
        time.sleep(30)"""
data = data.replace("src = args['input']", string)
out.write(data)
out.close()
data = data.replace("'Out')", "'In')")
data = data.replace("192.168.200.80:555", "192.168.200.79:554")
data = data.replace("'Out:", "'In:")
data = data.replace('[375, 200, 350, 275]', '[275, 200, 360, 245]')
in_.write(data)
in_.close()
