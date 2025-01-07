import os

path = os.environ['PATH']
print(path)
print(os.environ['PATH'])
os.environ['PATH'] = path + ':/ProphecyPlus/Marabou/Marabou_bld:/ProphecyPlus/Marabou/Marabou_bld/build:/ProphecyPlus/Marabou/Marabou_bld/build/bin'
print(os.environ['PATH'])


