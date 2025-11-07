import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_event_histogram(hist, xedges, yedges, allhist, title, iPlot,
        diff=None, var=None):
    '''
    Formats the 2D histogram plots.
    '''
    
    # have to check for nans. I am leaving the NaNs in until just before
    # plotting on purpose. It is useful to have them around to easily cut
    # out data.
    if(iPlot == 0): # Plotting a value that is not a difference
        cmap = 'Greys'
        max_val = np.max(hist[~np.isnan(hist)])
        min_val = np.min(hist[~np.isnan(hist)])
    if(iPlot == 1): # Plotting a value that is a difference
        cmap = 'bwr'
        mostpos = np.max(hist[~np.isnan(hist)])
        mostneg = np.min(hist[~np.isnan(hist)])
        max_val =  np.max( [np.abs(mostpos), np.abs(mostneg)] )
        min_val = -np.max( [np.abs(mostpos), np.abs(mostneg)] )


    fig, ax = plt.subplots(1,1, figsize=(12,6), facecolor='w', edgecolor='k')
    extents = (xedges.min(), xedges.max(),
            yedges.min(), yedges.max())
    imgplot = ax.imshow(hist.T, cmap=cmap, vmax=max_val, vmin=min_val,
            aspect='equal', origin='lower')

    if(iPlot == 1):
        ax.annotate(r'diff = {:.2e}'.format(diff),
                xy=(0.75,0.92), xycoords='axes fraction',
                fontsize=16, color='black')
        ax.annotate(r'var = {:.2e}'.format(var),
                xy=(0.75,0.82), xycoords='axes fraction',
                fontsize=16, color='black')

    draw_cell_borders(ax, allhist.T)

    xstep = xedges[1]-xedges[0]
    x_tick_locations = [i for i in range(1,len(xedges),2)]
    x_tick_labels = [int(xedges[i]+0.5*xstep) for i in x_tick_locations]

    ystep = yedges[1]-yedges[0]
    y_tick_locations = [i for i in range(len(yedges)-1)]
    y_tick_labels = [int(yedges[i]+0.5*ystep) for i in y_tick_locations]
    y_tick_labels[-1] = '>90'

    ax.set_xticks(x_tick_locations)
    ax.set_xticklabels(x_tick_labels, fontsize=14)
    ax.set_yticks(y_tick_locations)
    ax.set_yticklabels(y_tick_labels, fontsize=14)

    ax.set_xlabel('Distance from the net (feet)', fontsize=16, labelpad=12)
    ax.set_ylabel('Angle to the net (degrees)', fontsize=16, labelpad=12)
    ax.set_title(title, fontsize=18, pad=20)

    cb = fig.colorbar(imgplot)
    cb.set_label(r'$\Delta xG=(xG)_{\text{subset}} - (xG)_{\text{all data}}$',
            fontsize=18, labelpad=10)

    plt.show()

def draw_cell_borders(ax, hist):
    '''
    Draws borders in the 2D hist plots where data will not exist. Occurs for 
    combinations of distance & angle that area outside the arena.

    The histogram that should be used is the one for all events in the full dataset.
    '''

    [M,N] = hist.shape
    lw = 3

    for j in range(N): # along axis 1 in the histogram
        for i in range(M-1): # through axis 0
            if(hist[i,j] != 0 and hist[i+1,j] == 0 and i != M-1):
                ax.plot([j-0.5,j+0.5],[i+0.5,i+0.5],'k-', lw=lw) # Draw a line
            elif(hist[i,j] == 0 and hist[i+1,j] != 0 and i != M-1):
                ax.plot([j-0.5,j+0.5],[i+0.5,i+0.5],'k-', lw=lw)

    for i in range(M): # along axis 0 in the histogram
        for j in range(N-1): # through axis 1
            if(hist[i,j] != 0 and hist[i,j+1] == 0 and j != N-1):
                ax.plot([j+0.5,j+0.5],[i-0.5,i+0.5],'k-', lw=lw) # Draw a line
            elif(hist[i,j] == 0 and hist[i,j+1] != 0 and j != N-1):
                ax.plot([j+0.5,j+0.5],[i-0.5,i+0.5],'k-', lw=lw)


def create_rink(
    ax, 
    plot_half = False,
    board_radius = 28,
    alpha = 1,
):
    '''
    Plots some lines which correspond to the dimensions & design of an NHL
    ice hockey rink.

    Taken from https://github.com/gredelsheimer/RandomCode
    '''

    #Corner Boards
    ax.add_artist(mpl.patches.Arc((100-board_radius , (85/2)-board_radius),
        board_radius * 2, board_radius * 2 , theta1=0, theta2=89,
        edgecolor='Black', lw=4.5,zorder=0, alpha = alpha)) #Top Right
    ax.add_artist(mpl.patches.Arc((-100+board_radius+.1 , (85/2)-board_radius),
        board_radius * 2, board_radius * 2 ,theta1=90, theta2=180,
        edgecolor='Black', lw=4.5,zorder=0, alpha = alpha)) #Top Left
    ax.add_artist(mpl.patches.Arc((-100+board_radius+.1 , -(85/2)+board_radius-.1),
        board_radius * 2, board_radius * 2 ,theta1=180, theta2=270,
        edgecolor='Black', lw=4.5,zorder=0, alpha = alpha)) #Bottom Left
    ax.add_artist(mpl.patches.Arc((100-board_radius , -(85/2)+board_radius-.1), 
        board_radius * 2, board_radius * 2 ,theta1=270, theta2=360,
        edgecolor='Black', lw=4.5,zorder=0, alpha = alpha)) #Bottom Right

    #[x1,x2],[y1,y2]
    #Plot Boards 
    ax.plot([-100+board_radius,100-board_radius], [-42.5, -42.5],
            linewidth=4.5, color="Black",zorder=0, alpha = alpha) #Bottom
    ax.plot([-100+board_radius-1,100-board_radius+1], [42.5, 42.5],
            linewidth=4.5, color="Black",zorder=0, alpha = alpha) #Top
    ax.plot([-100,-100], [-42.5+board_radius, 42.5-board_radius],
            linewidth=4.5, color="Black",zorder=0, alpha = alpha) #Left
    ax.plot([100,100], [-42.5+board_radius, 42.5-board_radius],
            linewidth=4.5, color="Black",zorder=0, alpha = alpha) #Right

    #Goal Lines 
    adj_top = 5.8
    adj_bottom = 5.8
    ax.plot([89,89], [-42.5+adj_bottom, 42.5 - adj_top],
            linewidth=3, color="Red",zorder=-1, alpha = alpha)
    ax.plot([-89,-89], [-42.5+adj_bottom, 42.5 - adj_top],
            linewidth=3, color="Red",zorder=-1, alpha = alpha)

    #Plot Center Line
    ax.plot([0,0], [-42.5, 42.5], linewidth=3, color="Red",zorder=0, alpha = alpha)
    ax.plot(0,0, markersize = 6, color="Blue", marker = "o",
            zorder=0, alpha = alpha) #Center FaceOff Dots
    ax.add_artist(mpl.patches.Circle((0, 0), radius = 33/2, facecolor='none',
        edgecolor="Blue", linewidth=3,zorder=0, alpha = alpha)) #Center Circle

    #Zone Faceoff Dots
    ax.plot(69,22, markersize = 6, color="Red", marker = "o",zorder=0, alpha = alpha)
    ax.plot(69,-22, markersize = 6, color="Red", marker = "o",zorder=0, alpha = alpha)
    ax.plot(-69,22, markersize = 6, color="Red", marker = "o",zorder=0, alpha = alpha)
    ax.plot(-69,-22, markersize = 6, color="Red", marker = "o",zorder=0, alpha = alpha)

    #Zone Faceoff Circles
    ax.add_artist(mpl.patches.Circle((69, 22), radius = 15,
        facecolor='none', edgecolor="Red", linewidth=3,zorder=0, alpha = alpha)) 
    ax.add_artist(mpl.patches.Circle((69,-22), radius = 15,
        facecolor='none', edgecolor="Red", linewidth=3,zorder=0, alpha = alpha)) 
    ax.add_artist(mpl.patches.Circle((-69,22), radius = 15,
        facecolor='none', edgecolor="Red", linewidth=3,zorder=0, alpha = alpha)) 
    ax.add_artist(mpl.patches.Circle((-69,-22), radius = 15,
        facecolor='none', edgecolor="Red", linewidth=3,zorder=0, alpha = alpha)) 

    #Neutral Zone Faceoff Dots
    ax.plot(22,22, markersize = 6, color="Red", marker = "o",zorder=0, alpha = alpha)
    ax.plot(22,-22, markersize = 6, color="Red", marker = "o",zorder=0, alpha = alpha)
    ax.plot(-22,22, markersize = 6, color="Red", marker = "o",zorder=0, alpha = alpha)
    ax.plot(-22,-22, markersize = 6, color="Red", marker = "o",zorder=0, alpha = alpha)

    #Plot Blue Lines
    ax.plot([25,25], [-42.5, 42.5], linewidth=2, color="Blue",zorder=0, alpha = alpha)
    ax.plot([-25,-25], [-42.5, 42.5], linewidth=2, color="Blue",zorder=0, alpha = alpha)

    #Goalie Crease
    ax.add_artist(mpl.patches.Arc((89, 0), 8,8,theta1=90, theta2=270,
        facecolor="Blue", edgecolor='Red', lw=2,zorder=0, alpha = alpha))
    ax.add_artist(mpl.patches.Arc((-89, 0), 8,8, theta1=270, theta2=90,
        facecolor="Blue", edgecolor='Red', lw=2,zorder=0, alpha = alpha))

    #Goal
    ax.add_artist(mpl.patches.Rectangle((89, 0 - (6/2)), 40/12, 6,
        lw=2, color='Red',fill=False,zorder=0, alpha = alpha))
    ax.add_artist(mpl.patches.Rectangle((-89 - 2, 0 - (6/2)), 40/12, 6,
        lw=2, color='Red',fill=False,zorder=0, alpha = alpha))

    if plot_half == False:
        # Set axis limits
        ax.set_xlim(-101, 101)
        #ax.set_ylim(-43, 43)  

    elif plot_half == True:
        # Set axis limits
        ax.set_xlim(-0.5, 100.5)
        #ax.set_ylim(-43, 43) 


    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)


