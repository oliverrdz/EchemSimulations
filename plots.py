#!/usr/bin/python3

import matplotlib.pyplot as plt

def plotFormat():
	plt.xticks(fontsize = 14)
	plt.yticks(fontsize = 14)
	plt.grid()
	plt.tight_layout()

def plot(x, y, xlab, ylab, marker="-"):
	plt.plot(x, y, marker)
	plt.xlabel(xlab, fontsize = 18)
	plt.ylabel(ylab, fontsize = 18)
	plotFormat()
	plt.show()

def plot2(x1, y1, x2, y2, lab1, lab2, xlab, ylab, marker1="-", marker2 = "-", loc=1):
	plt.plot(x1, y1, marker1, label = lab1)
	plt.plot(x2, y2, marker2, label = lab2)
	plt.xlabel(xlab, fontsize = 18)
	plt.ylabel(ylab, fontsize = 18)
	plt.legend(loc = loc, fontsize = 14)
	plotFormat()
	plt.show()