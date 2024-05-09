import re
def second2str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)
class OffsetList(list):
    def __init__(self, seq=()):
        super(OffsetList, self).__init__(seq)
        self.offset = 0

    def min_index(self):
        return self.offset

    def max_index(self):
        return self.offset + len(self) - 1

    def __getitem__(self, item):
        return super(OffsetList, self).__getitem__(max(0, item - self.offset))

    def append(self, __object) -> None:
        super(OffsetList, self).append(__object)

    def pop(self):
        self.offset += 1
        super(OffsetList, self).pop(0)

    def clear(self) -> None:
        self.offset = 0
        super(OffsetList, self).clear()
zhPattern = re.compile(u'[\u4e00-\u9fa5|\d|a-z|A-Z]+')
def validFileName(filename):
    return True if zhPattern.fullmatch(filename) else False

if __name__ == '__main__':
    print(validFileName(""))
