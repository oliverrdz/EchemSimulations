#### Function that creates a potential sweep waveform
'''
    Copyright (C) 2020 Oliver Rodriguez
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
#### @author oliverrdz
#### https://oliverrdz.xyz

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
