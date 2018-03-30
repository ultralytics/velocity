import numpy as np
from bokeh import io, plotting, palettes, models
from bokeh.layouts import column, row


def plotresults(cam, im, P, S, B, bbox):
    # Bokeh Plotting
    io.reset_output()
    io.output_file('bokeh plots.html', title='bokeh plots')
    h, w = im.shape
    n = P.shape[2]  # number of images
    xn = list(range(0, n, 1))
    colors = bokeh_colors(palettes, n)
    colorbyframe = True

    # Setup tools
    hoverImage = models.HoverTool(tooltips=[('index', '$index'), ('(x, y)', '($x{(1.11)}, $y{(1.11)})')],
                                  point_policy='follow_mouse')
    hover = models.HoverTool(tooltips=[('index', '$index'), ('(x, y)', '(@x{(1.11)}, @y{(1.11)})')],
                             point_policy='snap_to_data', mode='vline')  # $x refers to mouse pos, @x to datapoint pos

    a = plotting.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(800 * w / h), plot_height=800,
                        x_axis_label='pixel (1 - ' + str(w) + ')', y_axis_label='pixel (1 - ' + str(h) + ')',
                        title=cam['filename'],
                        tools='box_zoom,pan,save,reset,wheel_zoom,crosshair',
                        active_scroll='wheel_zoom', active_inspect=None)
    a.add_tools(hoverImage)
    a.legend.click_policy = 'hide'

    # Plot image
    a.image(image=[np.flip(im, 0)], x=0, y=h, dw=w, dh=h, palette=palettes.Greys256)
    a.quad(top=[h], bottom=[0], left=[0], right=[w], alpha=0)  # clear rectange hack for tooltip image x,y

    # Plog bounding box
    a.quad(top=bbox[1] + bbox[3], bottom=bbox[1], left=bbox[0], right=bbox[0] + bbox[2], color=colors[1], line_width=2,
           alpha=.3)  # clear rectange hack for tooltip image x,y

    # Plot license plate outline
    a.patch(P[0, 0:4, 0], P[1, 0:4, 0], alpha=0.3, line_width=4)

    # Plot points
    if colorbyframe:
        for i in np.arange(n):
            a.circle(P[0, :, i], P[1, :, i], color=colors[i], legend='image ' + str(i), line_width=1)
            a.circle(P[2, :, i], P[3, :, i], color=colors[i], size=9, alpha=.5)
    else:
        a.circle(P[0,].ravel(), P[1,].ravel(), color=colors[0], legend='Points', line_width=2)
        a.circle(P[2,].ravel(), P[3,].ravel(), color=colors[1], legend='Reprojections', size=9, alpha=.7)

    # Plot 2 - 3d
    x, y, z = np.split(B[:, :3], 3, axis=1)
    b = plotting.figure(x_range=(-y.max() - 1, y.max() + 1), y_range=(0, x.max() + 1), plot_width=350, plot_height=300,
                        x_axis_label='Y (m)', y_axis_label='X (m)', title='Position',
                        tools='save,reset', active_inspect=hover)
    b.add_tools(hover)

    # Plot 3 - distance
    c = plotting.figure(x_range=(0, n), y_range=(0, round(S[1:, 7].max() + 1)), plot_width=350, plot_height=300,
                        x_axis_label='image', y_axis_label='distance (m)', title='Distance',
                        tools='save,reset', active_inspect=hover)
    c.add_tools(hover)

    # Plot 4 - speed
    d = plotting.figure(x_range=(0, n), y_range=(0, np.round_(S[1:, 8].max() + 10, -1)), plot_width=350,
                        plot_height=300,
                        x_axis_label='image', y_axis_label='speed (km/h)', title='Speed',
                        tools='save,reset', active_inspect=hover)
    d.add_tools(hover)

    # Circles plots 2-4
    if colorbyframe:
        for i in np.arange(n):
            b.circle(y[i], x[i], color=colors[i], line_width=2)
            c.circle(xn[i], S[i, 7], color=colors[i], line_width=2)
            d.circle(xn[i], S[i, 8], color=colors[i], line_width=2)
    else:
        b.circle(y.ravel(), x.ravel(), color=colors[0], line_width=2)
        c.circle(xn[1:], S[1:, 7], color=colors[0], line_width=2)
        d.circle(xn[1:], S[1:, 8], color=colors[0], line_width=2)

    # Plot lines
    a.multi_line(P[0,].tolist(), P[1,].tolist(), color='white', alpha=.7, line_width=1)

    # Show plot
    io.show(column(a, row(b, c, d)))  # open a browser


def bokeh_colors(bkpalettes, n):
    # https: // bokeh.pydata.org / en / latest / docs / reference / palettes.html
    # returns appropriate 10, 20 or 256 colors for plotting. n is the maximum required colors
    if n < 11:
        return bkpalettes.Category10[10]
    elif n < 21:
        return bkpalettes.Category20[20]
    else:
        return bkpalettes.Viridis256


def imshow(im, im2=None):
    # Bokeh Plotting
    io.reset_output()
    io.output_file('bokeh imshow.html', title='imshow')
    h, w = im.shape

    # Setup tools
    hoverImage = models.HoverTool(tooltips=[('index', '$index'), ('(x, y)', '($x{(1.11)}, $y{(1.11)})')],
                                  point_policy='follow_mouse')

    p = plotting.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(800 * w / h), plot_height=800,
                        x_axis_label='pixel (1 - ' + str(w) + ')', y_axis_label='pixel (1 - ' + str(h) + ')',
                        title='image', tools='box_zoom,pan,save,reset,wheel_zoom,crosshair',
                        active_scroll='wheel_zoom', active_inspect=None)
    p.add_tools(hoverImage)

    # Plot image
    p.image(image=[np.flip(im, 0)], x=0, y=h, dw=w, dh=h, palette=palettes.Greys256)
    p.quad(top=[h], bottom=[0], left=[0], right=[w], alpha=0)  # clear rectange hack for tooltip image x,y

    # Show plot
    if im2 is None:
        io.show(p)  # open a browser
    else:
        h, w = im2.shape
        q = plotting.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(800 * w / h), plot_height=800,
                            x_axis_label='pixel (1 - ' + str(w) + ')', y_axis_label='pixel (1 - ' + str(h) + ')',
                            title='image', tools='box_zoom,pan,save,reset,wheel_zoom,crosshair',
                            active_scroll='wheel_zoom', active_inspect=None)
        q.add_tools(hoverImage)

        # Plot image
        q.image(image=[np.flip(im2, 0)], x=0, y=h, dw=w, dh=h, palette=palettes.Greys256)
        q.quad(top=[h], bottom=[0], left=[0], right=[w], alpha=0)  # clear rectange hack for tooltip image x,y

        # Show plots
        io.show(column(p, q))  # open a browser
