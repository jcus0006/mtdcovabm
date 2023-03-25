class AccomGroup: #not useful/used
    def __init__(self):
        self.accomtypeids = []
        self.accomids = []
        self.roomids = []
        self.roomsizes = []
        self.cellindices = []

    def append(self, accomtypeid, accomid, roomid, roomsize, cellindex):
        self.accomtypeids.append(accomtypeid)
        self.accomids.append(accomid)
        self.roomids.append(roomid)
        self.roomsizes.append(roomsize)
        self.cellindices.append(cellindex)

