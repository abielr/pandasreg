import os
import subprocess
import uuid
import glob
import numpy as np
import pandas as pd
from pandasreg.rperiod import RPeriodIndex, RFrequency, RPeriod

def d(series, n=1):
      return series-series.shift(n)

def da(series, n=1):
      return (series-series.shift(n))*series.index.freq.periodicity

def dy(series, n=1):
      return (series-series.shift(n*series.index.freq.periodicity))

def dya(series, n=1):
      return (series-series.shift(n*series.index.freq.periodicity)) / n

def logd(series, n=1):
      return (np.log(series)-np.log(series.shift(n)))*100

def logda(series, n=1):
      return (np.log(series)-np.log(series.shift(n)))*series.index.freq.periodicity*100

def logdy(series, n=1):
      return (np.log(series)-np.log(series.shift(n*series.index.freq.periodicity)))*100

def logdya(series, n=1):
      return (np.log(series)-np.log(series.shift(n*series.index.freq.periodicity)))*100.0/n

def pc(series, n=1):
      return (series/series.shift(n)-1)*100

def pca(series, n=1):
      return ((series/series.shift(n))**(1.0*series.index.freq.periodicity/n)-1)*100

def pcy(series, n=1):
      return (series/series.shift(n*series.index.freq.periodicity)-1)*100

def pcya(series, n=1):
      return ((series/series.shift(n*series.index.freq.periodicity))**(1.0/n)-1)*100,

def x12(series):
    if series.index.freq.freqstr == "M":
        if len(series.values) < 36:
            raise ValueError("Must have at least three years of data")
        start = series.index[0].strftime("%Y.%m")
        period = 12
    elif series.index.freq.freqstr.startswith("Q"):
        if len(series.values) < 12:
            raise ValueError("Must have at least three years of data")
        quarter = str((series.index[0].to_datetime().month-1)/3+1)
        start = series.index[0].strftime("%Y")+"."+quarter
        period = 4
    else:
        return series # Can only do adjustment on monthly and quarterly data with X-12

    template = """
      series{
        title="Test"
        start=%s
        period=%d
        data=(
          %s
        )
        span=(%s,)
      }
      x11{
          print=(d11)
      }
    """ % (start, period, "\n".join([str(x) for x in series.values]), start)

    spcpath = os.path.dirname(__file__)+"/../../tsdata/tmp/"
    spcname = str(uuid.uuid4())
    spcfile = spcpath+spcname+".spc"
    with open(spcfile, "w") as f:
        f.write(template)

    subprocess.call([os.path.dirname(__file__)+"/../../../bin/x13as/x13as.exe", spcpath+spcname, "-Q", "-P", "-N", "-R"], 
        stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    with open(spcpath+spcname+".out") as f:
        divcount = 0
        line = f.readline()
        while divcount < 2:
            if line.startswith(" -----"):
                divcount += 1
            line = f.readline()

        values = []
        while line.find("AVGE") < 0:
            if line[2:6].isdigit():
                if series.index.freq.freqstr == "M":
                    values += line.strip().split()[1:]
                    line = f.readline()
                    values += line.strip().split()[:-1]
                else: # if quarterly
                    values += line.strip().split()[1:-1]
            line = f.readline()
        values = [float(value) for value in values]

    for filename in glob.glob(spcpath+"/"+spcname+"*"):
        os.remove(filename)

    return type(series)(values, index=series.index)