import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import optimize

# ---------------------- Routine for plots with histograms ------------- #

def make_all_ctr_hist_plots(df, imgdir,
        hist_SAT, hist_shots, hist_misses, hist_goals,
        dist_edges, angle_edges, hist0_SAT,
        titledict=None, column=None, column_value=None):
    '''
    A routine that calls the plot_hist_with_heatmap routine for the
    various events: SOG, misses, goals, SAT.
    '''

    # SAT: shot attempts = SOG + miss + goal
    if titledict is not None:
        plttitle = r"Distribution of SAT for " +\
            titledict[column+'-'+str(column_value)]
        imgstr = imgdir + column + '-' + str(column_value) +\
            '-SAT.png'
    else:
        plttitle = r"Distribution of SAT for reference dataset"
        imgstr = imgdir + 'ref-SAT.png'

    histdf = df.loc[df['event'].isin(['SHOT', 'MISS','GOAL'])]
    Nevents = len(histdf.index)
    plot_hist_with_heatmap(hist_SAT,
            dist_edges, angle_edges, hist0_SAT, plttitle,
            histdf['x'].values, histdf['y'].values, Nevents,
            imgstr=imgstr)


    # SOG: shots on goal
    if titledict is not None:
        plttitle = r"Distribution of SOG for " +\
            titledict[column+'-'+str(column_value)]
        imgstr = imgdir + column + '-' + str(column_value) + '-SOG.png'
    else:
        plttitle = r"Distribution of SOG for reference dataset"
        imgstr = imgdir + 'ref-SOG.png'

    histdf = df.loc[df['event'].isin(['SHOT'])]
    Nevents = len(histdf.index)
    plot_hist_with_heatmap(hist_shots,
            dist_edges, angle_edges, hist0_SAT, plttitle,
            histdf['x'].values, histdf['y'].values, Nevents,
            imgstr=imgstr)


    # Miss: shots that missed the net
    if titledict is not None:
        plttitle = r"Distribution of misses for " +\
            titledict[column+'-'+str(column_value)]
        imgstr = imgdir + column + '-' + str(column_value) + '-miss.png'
    else:
        plttitle = r"Distribution of misses for reference dataset"
        imgstr = imgdir + 'ref-misses.png'

    histdf = df.loc[df['event'].isin(['MISS'])]
    Nevents = len(histdf.index)
    plot_hist_with_heatmap(hist_misses,
            dist_edges, angle_edges, hist0_SAT, plttitle,
            histdf['x'].values, histdf['y'].values, Nevents,
            imgstr=imgstr)


    # shots resulting in a goal
    if titledict is not None:
        plttitle = r"Distribution of goals for " +\
            titledict[column+'-'+str(column_value)]
        imgstr = imgdir + column + '-' + str(column_value) + '-goal.png'
    else:
        plttitle = r"Distribution of goals for reference dataset"
        imgstr = imgdir + 'ref-goals.png'

    histdf = df.loc[df['event'].isin(['GOAL'])]
    Nevents = len(histdf.index)
    plot_hist_with_heatmap(hist_goals,
            dist_edges, angle_edges, hist0_SAT, plttitle,
            histdf['x'].values, histdf['y'].values, Nevents,
            imgstr=imgstr)




# ---------------------- Routine for plots with histograms ------------- #

def plot_hist_with_heatmap(hist, xedges1, yedges1,
        allhist, title,
        xdata, ydata,
        Nann, imgstr=None):
    '''
    Formats the 2D histogram plots.
    '''

    cmap = 'PuBu'
    max_val = np.max(hist[~np.isnan(hist)])
    min_val = np.min(hist[~np.isnan(hist)])

    fig, [ax0, ax1] = plt.subplots(2,1, figsize=(12,12),
            height_ratios=[1.7,1],
            facecolor='w', edgecolor='k')

    #ax0.plot(xdata,ydata, '.')

    # ---- ax0: Event contour map ontop of rink drawing --- #

    # draw the rink first
    create_rink(ax0, board_radius=28, plot_half=True)
    ax0.set_aspect('equal') # to draw rink with correct scale in x & y

    # Bin all events in x-y coordinates, on a grid with 1 ft spacing
    xedges0 = np.linspace(-0.5,100.5,102)
    yedges0 = np.linspace(-42.5,42.5,86)

    extents = (xedges0.min(), xedges0.max(),
            yedges0.min(), yedges0.max())
    [histxy, xe, ye] = np.histogram2d(xdata, ydata,
            bins=(xedges0,yedges0))

    # smooth the histogram using a gaussian filter
    histxy_smooth = gaussian_filter(histxy.T, sigma=0)

    # Need the centres of the hist bin egdes for call to contourf
    xcen0 = xedges0[:-1]+0.5*(xedges0[1]-xedges0[0])
    ycen0 = yedges0[:-1]+0.5*(yedges0[1]-yedges0[0])
    xm, ym = np.meshgrid(xcen0, ycen0)

    draw_hist_bins_on_rink(ax0, xedges1[1:], yedges1[:-1], alpha=0.6)

    # Create a contour plot, or heat map
    histmin = np.min(histxy_smooth); histmax = np.max(histxy_smooth);

    ctrf = ax0.contourf(xcen0, ycen0, histxy_smooth, alpha=0.85,
            cmap=cmap, vmin=histmin, vmax=histmax,
            levels = np.linspace(1,histmax,12))
    cb = fig.colorbar(ctrf)
    cb.set_label(r'Density of shots', fontsize=14, labelpad=10)
    cb.set_ticks([])
    


    # Flip this xaxis so the rink is drawn with increasing distance from
    # left to right, to match the histogram
    ax0.invert_xaxis()
    ax0.set_xticks([89, 69, 49, 25, 0])
    ax0.set_xticklabels(['0', '20', '40', '64', '90'], fontsize=11)

    ax0.set_xlabel(r'$x$ co-ordinate (feet)', fontsize=16, labelpad=12)
    ax0.set_ylabel(r'$y$ co-ordinate (feet)', fontsize=16, labelpad=12)
    ax0.xaxis.set_label_position('top')
    ax0.xaxis.tick_top()

    ax0.set_title(title, fontsize=18, pad=20)

    # ---- ax1: 2D histogram of event in distance-angle --- #

    # histogram is calculated outside of this function, passed in
    # via the hist variable

    plothist = np.copy(hist.T)
    # Assign NaN here so the bins are rendered as white in the plot
    # NaN is a good choice of value here -- these bins are outside the
    # O-zone, and thus cannot have data and should be ignored
    plothist[np.where(allhist.T==0)] = None  

    imgplot = ax1.imshow(plothist, cmap=cmap, vmax=max_val, vmin=min_val,
            aspect='equal', origin='lower')
    draw_cell_borders(ax1, allhist.T)

    ax1.annotate(r'N = {:d}'.format(Nann),
            xy=(0.72,0.92), xycoords='axes fraction',
            fontsize=16, color='black')

    xstep = xedges1[1]-xedges1[0]
    x_tick_locations = [i for i in range(1,len(xedges1[:-1]),2)]
    x_tick_labels = [int(xedges1[i]+0.5*xstep) for i in x_tick_locations]

    ystep = yedges1[1]-yedges1[0]
    y_tick_locations = [i for i in range(len(yedges1)-1)]
    y_tick_labels = [int(yedges1[i]+0.5*ystep) for i in y_tick_locations]
    y_tick_labels[-1] = '>90'

    ax1.set_xticks(x_tick_locations)
    ax1.set_xticklabels(x_tick_labels, fontsize=14)
    ax1.set_yticks(y_tick_locations)
    ax1.set_yticklabels(y_tick_labels, fontsize=14)

    ax1.set_xlabel('Distance from the net (feet)', fontsize=16, labelpad=12)
    ax1.set_ylabel('Angle to the net (degrees)', fontsize=16, labelpad=12)

    cb = fig.colorbar(imgplot)
    cb.set_label(r"Number of shots", fontsize=18, labelpad=10)

    if imgstr is None: # plotting the histogram directly
        plt.show()
    else: # saving the plot to a file at imgstr
        plt.savefig(imgstr, bbox_inches='tight')
        plt.close()


def plot_xG_deltaxG(xGhist, dxGhist, xedges, yedges, allhist, title,
        imgstr=None, ann_nums=None):
    '''
    Creates a 2-panel plot with two histograms: xG on the top,
    and a deltaxG on the bottom. Both of these histograms have
    to be passed in.
    '''

    # Setting up ticks, ticklabels, and extents for imshow

    xstep = xedges[1]-xedges[0]
    x_tick_locations = [i for i in range(1,len(xedges[:-1]),2)]
    x_tick_labels = [int(xedges[i]+0.5*xstep) for i in x_tick_locations]

    ystep = yedges[1]-yedges[0]
    y_tick_locations = [i for i in range(len(yedges)-1)]
    y_tick_labels = [int(yedges[i]+0.5*ystep) for i in y_tick_locations]
    y_tick_labels[-1] = '>90'

    extents = (xedges.min(), xedges.max(), yedges.min(), yedges.max())


    # Create a fig with two axes
    fig, [ax0, ax1] = plt.subplots(2,1, figsize=(9,9), sharex=True,
            facecolor='w', edgecolor='k')
    plt.subplots_adjust(hspace=0.05)
    

    # ---------- ax0: a plot of xG ------------------------------ #

    imgplot0 = ax0.imshow(xGhist.T, cmap='Greys',
            #vmax=max_val, vmin=min_val,
            aspect='equal', origin='lower')
    draw_cell_borders(ax0, allhist.T)

    # annotations for number of goals & SAT
    ax0.annotate(r'$N_{{goals}}$ = {:d}'.format(ann_nums[2]),
            xy=(0.68,0.92), xycoords='axes fraction',
            fontsize=16, color='black')
    ax0.annotate(r'$N_{{SAT}}$ = {:d}'.format(ann_nums[3]),
            xy=(0.68,0.82), xycoords='axes fraction',
            fontsize=16, color='black')
    ax0.annotate(r'$S\% [frac]$ = {:.3f}'.format(
            ann_nums[2]/ann_nums[3]),
            xy=(0.68,0.72), xycoords='axes fraction',
            fontsize=16, color='black')

    # set up colourbar
    cb0 = fig.colorbar(imgplot0, ax=ax0,
            fraction=0.12, pad=0.02, aspect=15, shrink=0.9)
    cb0.set_label(r'$xG=n_{{goals}}/n_{{SAT}}$',
            fontsize=18, labelpad=10)
    cb0.ax.yaxis.set_tick_params(labelsize=14)

    # set up ticks
    ax0.set_xticks([])
    ax0.set_yticks(y_tick_locations)
    ax0.set_yticklabels(y_tick_labels, fontsize=14)
    ax0.set_title(title, fontsize=18, pad=20)

    # ---------- ax1: a plot of Delta xG ------------------------- #

    # set up max & min values for the colourbar

    # Have to check for nans. I have left the NaNs in until just before
    # plotting on purpose. It is useful to have them around, to remind
    # myself that there shouldn't be data plotted @ the nan's
    mostpos = np.max(dxGhist[~np.isnan(dxGhist)])
    mostneg = np.min(dxGhist[~np.isnan(dxGhist)])
    max_val =  np.max( [np.abs(mostpos), np.abs(mostneg)] )
    min_val = -np.max( [np.abs(mostpos), np.abs(mostneg)] )

    imgplot1 = ax1.imshow(dxGhist.T, cmap='bwr', vmax=max_val, vmin=min_val,
            aspect='equal', origin='lower')
    draw_cell_borders(ax1, allhist.T)

    # annotations for "diff" and "variance" metrics I've calcualted
    ax1.annotate(r'diff = {:.2e}'.format(ann_nums[0]),
            xy=(0.68,0.92), xycoords='axes fraction',
            fontsize=16, color='black')
    ax1.annotate(r'var = {:.2e}'.format(ann_nums[1]),
            xy=(0.68,0.82), xycoords='axes fraction',
            fontsize=16, color='black')

    # set up colourbar
    cb1 = fig.colorbar(imgplot1, ax=ax1,
            fraction=0.12, pad=0.02, aspect=15, shrink=0.9)
    cb1.set_label(r'$\Delta xG=(xG)_{\text{subset}}-(xG)_{\text{all data}}$',
            fontsize=18, labelpad=10)
    cb1.ax.yaxis.set_tick_params(labelsize=14)

    # set up ticks
    ax1.set_xticks(x_tick_locations)
    ax1.set_xticklabels(x_tick_labels, fontsize=14)
    ax1.set_yticks(y_tick_locations)
    ax1.set_yticklabels(y_tick_labels, fontsize=14)

    # ax1, the lower axis, gets the xlabel
    ax1.set_xlabel('Distance from the net (feet)', fontsize=16, labelpad=12)


    # add a blank axis to wrap whole fig, just to add ylabel to this axis
    ax = fig.add_subplot(111, frameon=False)
    ax.set_ylabel('Angle to the net (degrees)', fontsize=16, labelpad=50)
    ax.set_xticks([])
    ax.set_yticks([])

    if imgstr is None: # plotting the histogram directly to screen
        plt.show()
    else: # saving the plot to a file at imgstr
        plt.savefig(imgstr, bbox_inches='tight')
        plt.close()


def plot_single_histogram(hist, xedges, yedges, allhist, title, ann_nums,
        imgstr=None):
    '''
    Formats a single 2D histogram plot.
    '''
    # Have to check for nans. I have left the NaNs in until just before
    # plotting on purpose. It is useful to have them around, to remind
    # myself that there shouldn't be data plotted @ the nan's

    cmap = 'Greys'
    max_val = np.max(hist[~np.isnan(hist)])
    min_val = np.min(hist[~np.isnan(hist)])


    fig, ax = plt.subplots(1,1, figsize=(11,5),
            facecolor='w', edgecolor='k')
    extents = (xedges.min(), xedges.max(),
            yedges.min(), yedges.max())

    imgplot = ax.imshow(hist.T, cmap=cmap, vmax=max_val, vmin=min_val,
            aspect='equal', origin='lower')

    draw_cell_borders(ax, allhist.T)

    ax.annotate(r'$N_{{goals}}$ = {:d}'.format(ann_nums[0]),
            xy=(0.72,0.92), xycoords='axes fraction',
            fontsize=16, color='black')
    ax.annotate(r'$N_{{SAT}}$ = {:d}'.format(ann_nums[1]),
            xy=(0.72,0.82), xycoords='axes fraction',
            fontsize=16, color='black')
    ax.annotate(r'$S\% [frac]$ = {:.3f}'.format(
            ann_nums[0]/ann_nums[1]),
            xy=(0.72,0.72), xycoords='axes fraction',
            fontsize=16, color='black')


    xstep = xedges[1]-xedges[0]
    x_tick_locations = [i for i in range(1,len(xedges[:-1]),2)]
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


    # set up colourbar
    cb = fig.colorbar(imgplot, ax=ax,
            fraction=0.12, pad=0.02, aspect=15, shrink=0.9)
    cb.set_label(r'$xG=n_{{goals}}/n_{{SAT}}$',
            fontsize=18, labelpad=10)
    cb.ax.yaxis.set_tick_params(labelsize=14)


    if imgstr is None: # plotting the histogram directly to screen
        plt.show()
    else: # saving the plot to a file at imgstr
        plt.savefig(imgstr, bbox_inches='tight')
        plt.close()


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



# -------- Routine for details in drawing hockey rink plots ------------- #

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
        ax.set_ylim(-43, 43)  

    elif plot_half == True:
        # Set axis limits
        ax.set_xlim(-0.5, 100.5)
        ax.set_ylim(-43, 43) 


    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)



def draw_hist_bins_on_rink(ax, r_bins, theta_bins,
        lw=1.5, ls='--', alpha=0.5, zorder=-10):
    '''
    Adds lines to the rink representing the distance-angle bins used
    in the 2D histograms.

    theta_bins must be in degrees, and have no number larger than 90.
    r_bins must be in feet, to match the units of the rink dimensions.
    '''

    # Relevant NHL rink dimensions, all units are feet

    rink_length = 200
    rink_width = 85    
    board_radius = 28
    # Location of the centre of the net (on the goal line)
    centre_net_x = 89; centre_net_y = 0;
    x_blue_line = 25
    deltaX_goal_boards = 11 # distance from goal line to end boards
    dist_blue_line = centre_net_x - x_blue_line

    # Two "critical" radii to determine intersections for the bin arcs
    y_board_corner = 0.5*rink_width - board_radius
    yc1 = y_board_corner
    xc1 = deltaX_goal_boards
    rc1 = np.sqrt(xc1**2 + yc1**2)

    x_board_corner = 0.5*rink_length - board_radius
    xc2 = centre_net_x - x_board_corner
    yc2 = 0.5*rink_width
    rc2 = np.sqrt(xc2**2 + yc2**2)

    # Draw arcs for the bin edges associated with distance (or radius)
    for r in r_bins:    

        # arc intersects the side boards and the blue line
        if (r > dist_blue_line):
            angle1 = (180/np.pi)*np.arccos(dist_blue_line/r)
            angle2 = (180/np.pi)*np.arcsin((0.5*rink_width)/r)

            # top part of the arc
            ax.add_artist(mpl.patches.Arc(
                (centre_net_x, centre_net_y), 2*r, 2*r,
                theta1=-angle2, theta2=-angle1, angle=180,
                edgecolor='Black', lw=lw, ls=ls,
                alpha=alpha, zorder=zorder))

            # bottom part of the arc
            ax.add_artist(mpl.patches.Arc(
                (centre_net_x, centre_net_y), 2*r, 2*r,
                theta1=angle1, theta2=angle2, angle=180,
                edgecolor='Black', lw=lw, ls=ls,
                alpha=alpha, zorder=zorder))


        # arc intersects both side boards
        elif(r > rc2 and r <= dist_blue_line):
            xt = np.sqrt( r**2 - (0.5*rink_width)**2 )
            yt = 0.5*rink_width
            anglet = np.arctan2(yt, xt) * (180/np.pi) # degrees
            ax.add_artist(mpl.patches.Arc(
                (centre_net_x, centre_net_y), 2*r, 2*r,
                theta1=-anglet, theta2=anglet, angle=180,
                edgecolor='Black', lw=lw, ls=ls,
                alpha=alpha, zorder=zorder))

        # arc intersects the corner
        elif(r < rc2 and r > rc1):

            def corner_intersect(x):
                C1 = r**2
                C2 = board_radius**2
                C3 = centre_net_x - (0.5*rink_length-board_radius)
                C4 = 0.5*rink_width - board_radius #- 10

                #print(C3,C4)

                ysolv = np.sqrt(C1-x**2) - (np.sqrt(C2 - (x+C3)**2) + C4)
                return ysolv

            # Newton's method would not converge; I suspect because
            # the function to solve diverges quickly near the solution
            #xt = optimize.newton(corner_intersect, 11)

            # bisect needs an interval: solution is guaranteed to be
            # within the interval [-radius of arc, end boards]
            xt = optimize.bisect(corner_intersect, -r, deltaX_goal_boards)

            anglet = (180/np.pi)*np.arcsin(xt/r)
            ax.add_artist(mpl.patches.Arc(
                (centre_net_x, centre_net_y), 2*r, 2*r,
                theta1=-(90+anglet), theta2=(90+anglet), angle=180,
                edgecolor='Black', lw=lw, ls=ls,
                alpha=alpha, zorder=zorder))


        # arc intersects below the corner boards
        elif(r < rc1):

            # arc intersects the end boards
            if ( r > deltaX_goal_boards):
                theta_gb = (180/np.pi)*np.arcsin( deltaX_goal_boards/r )
                ax.add_artist(mpl.patches.Arc(
                    (centre_net_x, centre_net_y), 2*r, 2*r,
                    theta1=-(90+theta_gb), theta2=(90+theta_gb), angle=180,
                    edgecolor='Black', lw=lw, ls=ls,
                    alpha=alpha, zorder=zorder))
 
            # arc intersects no boards; draw a full circle
            if ( r <= deltaX_goal_boards):
                ax.add_artist(mpl.patches.Arc(
                    (centre_net_x, centre_net_y), 2*r, 2*r,
                    theta1=-180, theta2=180, angle=180,
                    edgecolor='Black', lw=lw, ls=ls,
                    alpha=alpha, zorder=zorder))
      

    # Angle to determine whether this line intersects with the 
    # centre line or the boards
    xnb = centre_net_x - x_blue_line
    thetac1 = (180/np.pi)*np.arctan2( (0.5*rink_width), xnb )

    thetac2 = 90 - (180/np.pi)*np.arctan2(
        centre_net_x - (0.5*rink_length - board_radius), 0.5*rink_width)
    # Draw lines for the bin edges associated with angle
    for th in theta_bins:
        if (th <= thetac1): # intersects the blue line
            #rmax = np.max(r_bins)
            #yt = rmax * np.sin(th*(np.pi/180) )
            #xt = rmax * np.cos(th*(np.pi/180) )
            yi = dist_blue_line*np.tan(th*(np.pi/180))
            xi = x_blue_line

            # plot two straight lines, for pos & neg angles
            for anglesign in [-1, 1]:
                ax.plot( [xi, centre_net_x],
                        [yi*anglesign, centre_net_y],
                    color='Black', lw=lw, ls=ls,
                    alpha=alpha, zorder=zorder)

        #elif (th > thetac1): # intersects boards
        elif (th > thetac1 and th <= thetac2): # intersects boards
            xt = (0.5*rink_width)/np.tan(th*(np.pi)/180)
            xi = centre_net_x - xt

            # plot two straight lines, for pos & neg angles
            for anglesign in [-1,1]:
                ax.plot( [xi, centre_net_x],
                        [anglesign*0.5*rink_width, centre_net_y],
                    color='Black', lw=lw, ls=ls,
                    alpha=alpha, zorder=zorder)

        
        elif (th > thetac2): # intersects corner
            # Need to solve a quadratic equation
            centre_circle_x = 0.5*rink_length - board_radius
            centre_circle_y = 0.5*rink_width  - board_radius
            aangle = (np.pi/180) * (90-th)
 
            qC = centre_net_x - centre_circle_x - \
                    centre_circle_y*np.tan(aangle)
            aC = np.square(np.tan(aangle)) + 1
            bC = -2.0*np.tan(aangle)*qC
            cC = np.square(qC) - np.square(board_radius)

            y1 = ( -bC + np.sqrt(bC**2 - 4*aC*cC) ) / (2*aC)
            y2 = ( -bC - np.sqrt(bC**2 - 4*aC*cC) ) / (2*aC)
            # I want the positive solution
            ycc = np.max([y1,y2]) # distance from corner radius circle
            xcc = np.sqrt(board_radius**2 - ycc**2) # x-dist

            yi = centre_circle_y + ycc
            xi = centre_circle_x + xcc
            #print(xi, yi, '\n')

            # plot two straight lines, for pos & neg angles
            for anglesign in [-1,1]:
                ax.plot( [xi, centre_net_x],
                        [yi*anglesign, centre_net_y],
                    color='Black', lw=lw, ls=ls,
                    alpha=alpha, zorder=zorder)


# ---------------------- Scatter plots ------------------------------- #

def conditions_plot(conds, condvals, xdata, ydata, yavg, iPlot,
        title='', imgstr=None):
    """
    A scatter plot exploring summarized metrics to compare how
    subsets of data compare to the full data set.
    """

    fig, ax = plt.subplots(1,1, figsize=(9,6),
            facecolor='w', edgecolor='k')


    if(iPlot == 1): # Analysis #1
        xmin = -0.065; xmax = 0.082;
        ymin = -0.04 ; ymax = 0.12 ;
    else:
        xmin = None; xmax = None;
        ymin = None; ymax = None;


    # Horizontal & vertical lines through (0,0)
    ax.plot([xmin, xmax], [0.0  , 0.0  ], 'k--', alpha=0.3)
    ax.plot([0.0 , 0.0 ], [ymin , ymax ], 'k--', alpha=0.3)

    # Different conditions, e.g. shot type, team strength are in conds
    for i in range(len(conds)):
        ydata[i] -= yavg # subtract off an average from the ydata set

        # plot the data points
        # xdata[i] and ydata[i] are each 1D arrays
        ax.plot(xdata[i], ydata[i], '.')

        cond = conds[i];
        for anni in range(len(xdata[i])):
            val = condvals[i][anni] # Each value for each condition

            # x,y co-ordinates for annotating text
            annx = xdata[i][anni]+0.0007
            anny = ydata[i][anni]+0.0007

            # Manual shifts to avoid overlapping text
            if(iPlot == 1): # Analysis #1
                if(cond=='type' and val=='TIP'):
                    annx -= 0.0122
                    anny -= 0.0056
                if(cond=='PlayingStrength' and val=='5v3'):
                    annx -= 0.028
                    anny -= 0.0

            ax.annotate( str(conds[i])+':'+str(condvals[i][anni]),
                xy=(annx, anny), xycoords='data',
                fontsize=12, color='black')

    #ax.set_yscale('log')

    ax.set_xlabel(r'Weighted $\Delta$xG', fontsize=16, labelpad=12)
    ax.set_ylabel(r'$S\%_{{subset}}-S\%_{{all}}$',
            fontsize=16, labelpad=12)
    ax.set_title(title, fontsize=18, pad=20)


    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if imgstr is None: # plotting the histogram directly
        plt.show()
        #plt.close()
    else: # saving the plot to a file at imgstr
        plt.savefig(imgstr, bbox_inches='tight')
        plt.close()



# ---------------------- Book-keeping dictionaries ----------------- #

# For plot titles for the various conditions

titledict = {
        'bReb-0':r"non-rebound shots",
        'bReb-1':r"rebound shots",

        'type-WRIST':r"wrist shots",
        'type-SLAP' :r"slap shots",
        'type-TIP'  :r"tip shots",
        'type-SNAP' :r"snap shots",
        'type-WRAP' :r"wrap around shots",
        'type-BACK' :r"backhand shots",
        'type-DEFL' :r"deflected shots",

        'bPlayoffs-0':r"regular season games",
        'bPlayoffs-1':r"playoff games",
        'bForwardPlayer-0':r"shots taken by defence",
        'bForwardPlayer-1':r"shots taken by forwards",

        'bOffWing-0':r"shots from strong side",
        'bOffWing-1':r"shot from off wing",

        'anglesign-1' :r"shots from the goalie's left",
        'anglesign--1':r"shots from the goalie's right",

        'PlayingStrength-5v5':r"5v5",
        'PlayingStrength-4v5':r"4v5",
        'PlayingStrength-5v4':r"5v4",
        'PlayingStrength-6v5':r"6v5",
        'PlayingStrength-5v3':r"5v3",
        'PlayingStrength-3v3':r"3v3",
        'PlayingStrength-4v4':r"4v4"
}
