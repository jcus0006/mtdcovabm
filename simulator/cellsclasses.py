from enum import IntEnum

class CellType(IntEnum):
    Household = 0
    Workplace = 1 
    Accommodation = 2 
    Hospital = 3 
    Entertainment = 4 
    School = 5 
    Classroom = 6
    Institution = 7
    Transport = 8
    Religion = 9
    Airport = 10

class SimCellType(IntEnum):
    Residence = 0
    Workplace = 1 
    School = 2 
    Community = 3 