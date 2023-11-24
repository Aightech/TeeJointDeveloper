import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
import svgwrite

# # Radius1 and length of the tube
R1 = 305 #mm
R2 = 305 #mm
L = 2000 #mm
phi = np.pi/4
resolution = 500
theta2 = np.linspace(0, 2 * np.pi, resolution)
theta1 = np.linspace(0, 2 * np.pi, resolution)
x1 = np.zeros(resolution)
y1 = np.zeros(resolution)
z1 = np.zeros(resolution)
x2 = np.zeros(resolution)
y2 = np.zeros(resolution)
z2 = np.zeros(resolution)
l11 = np.zeros(resolution)
l12 = np.zeros(resolution)
l21 = np.zeros(resolution)
l22 = np.zeros(resolution)
grid1 = []
grid2 = []
p = 10

def rotate_tube(x, y, z, phi):
    # Rotate the tube around the x axis
    x, y, z = x, y * np.cos(phi) - z * np.sin(phi), y * np.sin(phi) + z * np.cos(phi)

    return x, y, z

def tube(R, L):
    # Create a grid of points in the tube
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, L, 10)
    u, v = np.meshgrid(u, v)

    # Parametric equations for the tube
    x = R * np.cos(u)
    y = R * np.sin(u)
    z = v

    return x, y, z

def draw_tube(ax,x,y,z, color):
    #plot all the points
    ax.scatter(x, y, z, c=color, marker='o', s=1)


def compute_development():
    global R1, R2, L, phi, theta1, theta2, l11, l12, l21, l22, grid1, grid2, x1, y1, z1, x2, y2, z2, p
    R1 = float(entryRadius1.get())
    R2 = float(entryRadius2.get())
    if R1 < R2:
        R1, R2 = R2, R1
    L = 4*R1
    phi = float(entryAngle.get())*np.pi/180
    theta1[:] = np.arccos(R2/R1*np.cos(theta2))

    # Tube 1
    x1, y1, z1 = tube(R1, L)
    x, y, z = tube(R1, -L)
    x1 = np.concatenate((x, x1), axis=0)
    y1 = np.concatenate((y, y1), axis=0)
    z1 = np.concatenate((z, z1), axis=0)

    # Tube 2
    x2, y2, z2 = tube(R2, L)
    x2, y2, z2 = rotate_tube(x2, y2, z2, phi)

    #calculate the development of the tube
    l11[:] = (R1*np.sin(np.arccos(R2/R1*np.cos(theta2)))*np.cos(phi) - R2*np.sin(theta2))/np.sin(phi)
    l12[:] = (R1*np.sin(np.arccos(R2/R1*np.cos(theta2)))*np.cos(phi) + R2*np.sin(theta2))/np.sin(phi)
    l21[:] = R1*np.sin(np.arccos(R2/R1*np.cos(theta2)))*np.sin(phi) + l11[:]*np.cos(phi)
    l22[:] = R1*np.sin(np.arccos(R2/R1*np.cos(theta2)))*np.sin(phi) + l12[:]*np.cos(phi)

    #calculate the grids
    grid1 = []
    N = theta1.shape[0]
    ystart = l11[N//4]
    ymid = l11[N//2]
    yend = l11[N//4*3]
    xstart = R1*theta1[0]
    xmid = R1*theta1[N//4]
    xend = R1*theta1[N//2]
    d = min(yend-ystart, xend-xstart)
    p = 10**np.floor(np.log10(d))#get the closesst power of 10 under d
    nb_rows = int((yend-ystart)//p)+1
    nb_cols = int((xend-xstart)//p)+1
    for i in range(1,nb_rows):
        #find l11 index where equal to ystart+i*p
        idx = []
        eps = 0.001
        while len(idx) < 2:
            idx = np.argwhere((np.abs(l11-ystart-i*p))<eps)
            eps += 0.01
        idx = np.reshape(idx, -1)
        grid1.append([[R1*theta1[idx[0]], R1*theta1[idx[1]]], [ystart+i*p, ystart+i*p]])
    for i in range(1,nb_cols):
        #find R1*theta1 index where equal to xstart+i*p
        idx = []
        eps = 0.001
        while len(idx) < 2:
            idx = np.argwhere((np.abs(R1*theta1-xstart-i*p))<eps)
            eps += 0.01
        idx = np.reshape(idx, -1)
        grid1.append([[xstart+i*p, xstart+i*p], [l11[idx[0]], l11[idx[1]]]])

    grid2 = []
    xstart = R2*theta2[0]
    xend = R2*theta2[-1]
    ystart = np.min(l21)
    yend = np.max(l21)
    nb_rows = int((yend-ystart)//p)+1
    nb_cols = int((xend-xstart)//p)+1
    for i in range(1,nb_rows):
        idx = np.reshape(np.argwhere((l21-ystart-i*p)<0), -1)
        if idx.shape[0] > 0:
            i1 = idx[0]
            i2 = idx[0]
            for j in range(1,idx.shape[0]):
                if idx[j] - idx[j-1] == 1:
                    i2 = idx[j]
                else:
                    grid2.append([[R2*theta2[i1], R2*theta2[i2]], [ystart+i*p, ystart+i*p]])
                    i1 = idx[j]
                    i2 = idx[j]
            if i1 != i2:
                grid2.append([[R2*theta2[i1], R2*theta2[i2]], [ystart+i*p, ystart+i*p]])
    for i in range(1,nb_cols):
        y = l21[np.argmin(np.abs(R2*theta2-xstart-i*p))]
        grid2.append([[xstart+i*p, xstart+i*p], [y,yend]])



def on_plot_click():
    global R1, R2, L, phi, theta1, theta2, l11, l12, l21, l22, grid1, grid2
    compute_development()

    # # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # x3 = R2 * np.cos(theta2)
    # y3 = R2 * np.sin(theta2)*np.cos(phi) - l22[:]*np.sin(phi)
    # z3 = R2 * np.sin(theta2)*np.sin(phi) + l22[:]*np.cos(phi)
    x4 = R1 * np.cos(theta1+np.pi)
    y4 = R1 * np.sin(theta1+np.pi)
    z4 = l11[:]

    #ax.plot(x3, y3, z3, 'k')
    ax.plot(x4, y4, z4, 'k')
    draw_tube(ax,x1,y1,z1,'b')
    draw_tube(ax,x2,y2,z2,'r')

    # Set labels and show plot
    plt.title(f'Tube with Radius1 = {R1}, Radius2 = {R2} and Length = {L}')
    #set the axis to be equal
    ax.set_xlim([-L, L])
    ax.set_ylim([-L, L])
    ax.set_zlim([-L, L])

    fig = plt.figure()
    ax1 = fig.add_subplot(212)
    ax1.plot(R2*theta2, l21, 'b')
    for i in range(len(grid2)):
        ax1.plot(grid2[i][0], grid2[i][1], 'k')
    
    ax2 = fig.add_subplot(211)
    ax2.plot(R1*theta1, l11, 'r')
    for i in range(len(grid1)):
        ax2.plot(grid1[i][0], grid1[i][1], 'k')    
    ax2.set_xlim([R2*theta2[0], R2*theta2[-1]])

    plt.show()

def on_save_click():
    global R1, R2, phi, theta1, theta2, l11, l12, l21, l22, grid1, grid2, p
    compute_development()

    file_name = "tube" + str(R1) + "_"+ str(R2) + "_" + str(phi*180/np.pi)

    #save the development of the tube in 2 svg files using the svgwrite library in mm
    dwg1 = svgwrite.Drawing(file_name + '_1.svg', profile='tiny', size=(str(R1*theta1[-1])+'mm', str(np.max(l11))+'mm'), viewBox=('0 0 ' + str(R1*theta1[-1]) + ' ' + str(np.max(l11))))
    dwg1.add(dwg1.polyline(points=[(R1*theta1[i], l11[i]) for i in range(theta1.shape[0])], stroke='red', fill='none'))
    grid1 = np.array(grid1)
    for i in range(len(grid1)):
        g = np.array(grid1[i])
        dwg1.add(dwg1.polyline(points=[(grid1[i][0][j], grid1[i][1][j]) for j in range(2)], stroke='black', fill='none'))
    dwg1.add(dwg1.text(str(p), insert=(R1*theta1[theta1.shape[0]//4], l11[theta1.shape[0]//2]), fill='black'))
    dwg1.save()

    dwg2 = svgwrite.Drawing(file_name + '_2.svg', profile='tiny', size=(str(R2*theta2[-1])+'mm', str(np.max(l21))+'mm'), viewBox=('0 0 ' + str(R2*theta2[-1]) + ' ' + str(np.max(l21))))
    dwg2.add(dwg2.polyline(points=[(R2*theta2[i], l21[i]) for i in range(theta2.shape[0])], stroke='blue', fill='none'))
    grid2 = np.array(grid2)
    for i in range(len(grid2)):
        dwg2.add(dwg2.polyline(points=[(grid2[i][0][j], grid2[i][1][j]) for j in range(2)], stroke='black', fill='none'))
    dwg2.add(dwg2.text(str(p), insert=(R2*theta2[theta2.shape[0]//2], np.mean(l21)), fill='black'))
    dwg2.save()
    print("Saved to " + file_name + "_1.svg and " + file_name + "_2.svg")
    



root = tk.Tk()
root.title("Tube development")
w_width = 300
w_height = 150
root.geometry(f"{w_width}x{w_height}")

frameTitle = tk.Frame(root)
frameTitle.pack(side=tk.TOP, fill=tk.X, expand=True)
labelTitle = tk.Label(frameTitle, text="Development of the tube")
labelTitle.pack(side=tk.TOP, fill=tk.X, expand=True)

frameRadius1 = tk.Frame(root)
frameRadius1.pack(side=tk.TOP)
labelRadius1 = tk.Label(frameRadius1, text="Radius1(mm): ")
labelRadius1.pack(side=tk.LEFT)
entryRadius1 = tk.Entry(frameRadius1)
entryRadius1.insert(0, str(R1))
entryRadius1.pack(side=tk.LEFT)

frameRadius2 = tk.Frame(root)
frameRadius2.pack(side=tk.TOP)
labelRadius2 = tk.Label(frameRadius2, text="Radius2(mm): ")
labelRadius2.pack(side=tk.LEFT)
entryRadius2 = tk.Entry(frameRadius2)
entryRadius2.insert(0, str(R2))
entryRadius2.pack(side=tk.LEFT)

frameAngle = tk.Frame(root)
frameAngle.pack(side=tk.TOP)
labelAngle = tk.Label(frameAngle,   text="Angle(deg):   ")
labelAngle.pack(side=tk.LEFT)
entryAngle = tk.Entry(frameAngle)
entryAngle.insert(0, str(phi*180/np.pi))
entryAngle.pack(side=tk.LEFT)

frameButton = tk.Frame(root)
frameButton.pack(side=tk.TOP)
button = tk.Button(frameButton, text="Plot", command=on_plot_click)
button.pack(side=tk.LEFT)
button = tk.Button(frameButton, text="Save", command=on_save_click)
button.pack(side=tk.LEFT)


root.mainloop()





