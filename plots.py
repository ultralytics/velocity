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
    colors = bokeh_colors(n)
    colorbyframe = True

    # Setup tools
    hoverImage = models.HoverTool(tooltips=[('(x, y)', '$x{1.11}, $y{1.11}')], point_policy='follow_mouse')

    # Plot image
    a = plotting.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(600 * w / h), plot_height=600,
                        x_axis_label='pixel (1 - ' + str(w) + ')', y_axis_label='pixel (1 - ' + str(h) + ')',
                        title=cam['filename'],
                        tools='box_zoom,pan,save,reset,wheel_zoom',
                        active_scroll='wheel_zoom', active_inspect=None)
    a.add_tools(hoverImage)
    a.legend.click_policy = 'hide'
    a.image(image=[np.flip(im, 0)], x=0, y=h, dw=w, dh=h, palette=palettes.Greys256)
    a.quad(top=[h], bottom=[0], left=[0], right=[w], alpha=0)  # clear rectange hack for tooltip image x,y

    # Plog bounding box
    a.quad(top=bbox[3], bottom=bbox[2], left=bbox[0], right=bbox[1], color=colors[1], line_width=2, alpha=.3)

    # Plot license plate outline
    a.patch(P[0, 0:4, 0], P[1, 0:4, 0], alpha=0.3, line_width=4)

    # Plot points
    if colorbyframe:
        # for i in list((0, n - 1)):  # plot first and last
        for i in range(n):  # plot all
            a.circle(P[0, :, i], P[1, :, i], color=colors[i], legend='image ' + str(i), line_width=1)
            a.circle(P[2, :, i], P[3, :, i], color=colors[i], size=8, alpha=.6)
    else:
        a.circle(P[0].ravel(), P[1].ravel(), color=colors[0], legend='Points', line_width=2)
        a.circle(P[2].ravel(), P[3].ravel(), color=colors[1], legend='Reprojections', size=8, alpha=.6)

    # Plot 2 - 3d
    x, y, z = np.split(B[:, :3], 3, axis=1)
    b = plotting.figure(plot_width=350, plot_height=300,
                        x_axis_label='Y (m)', y_axis_label='X (m)', title='Position',
                        tools='save,reset,hover', active_inspect='hover')

    # Plot 3 - distance
    c = plotting.figure(x_range=(0, n), y_range=(0, round(S[1:, 7].max())), plot_width=350, plot_height=300,
                        x_axis_label='image', y_axis_label='distance (m)',
                        title='Distance = %.2fm in %.3fs' % (S[-1, 7], S[-1, 5] - S[0, 5]),
                        tools='save,reset,hover', active_inspect='hover')

    # Plot 4 - speed
    d = plotting.figure(x_range=(0, n), y_range=(0, S[1:, 8].max() + 1), plot_width=350,
                        plot_height=300,
                        x_axis_label='image', y_axis_label='speed (km/h)',
                        title='Speed = %.2f +/- %.2f km/h' % (S[1:, 8].mean(), S[1:, 8].std()),
                        tools='save,reset,hover', active_inspect='hover')

    # Circles plots 2-4
    b.circle(0, 0, color=colors[-1], line_width=15)
    if colorbyframe:
        for i in range(n):
            b.circle(x[i], z[i], color=colors[i], line_width=2)
            c.circle(xn[i], S[i, 7], color=colors[i], line_width=2)
            d.circle(xn[i], S[i, 8], color=colors[i], line_width=2)
    else:
        b.circle(x.ravel(), z.ravel(), color=colors[0], line_width=2)
        c.circle(xn[1:], S[1:, 7], color=colors[0], line_width=2)
        d.circle(xn[1:], S[1:, 8], color=colors[0], line_width=2)

    # Plot lines
    a.multi_line(P[0].tolist(), P[1].tolist(), color='white', alpha=.7, line_width=1)

    # Show plot
    io.show(column(a, row(b, c, d)))  # open a browser


def bokeh_colors(n):
    # https: // bokeh.pydata.org / en / latest / docs / reference / palettes.html
    # returns appropriate 10, 20 or 256 colors for plotting. n is the maximum required colors
    if n < 11:
        return palettes.Category10[10]
    elif n < 21:
        return palettes.Category20[20]
    elif n < 256:
        g = []
        for i in np.linspace(0, 255, n, dtype=int):
            g.append(palettes.Viridis256[i])
        return g
    else:
        return palettes.Viridis256


def imshow(im, im2=None, p1=None, p2=None):
    # Bokeh Plotting
    io.reset_output()
    io.output_file('bokeh imshow.html', title='imshow')
    colorImage = (len(im.shape) == 3)
    if colorImage:
        h, w, _ = im.shape
    else:
        h, w = im.shape

    # Setup tools
    hoverImage = models.HoverTool(tooltips=[('index', '$index'), ('(x, y)', '($x{(1.11)}, $y{(1.11)})')],
                                  point_policy='follow_mouse')

    p = plotting.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(800 * w / h), plot_height=800,
                        x_axis_label='pixel (1 - ' + str(w) + ')', y_axis_label='pixel (1 - ' + str(h) + ')',
                        title='image', tools='box_zoom,pan,save,reset,wheel_zoom,crosshair',
                        active_scroll='wheel_zoom', active_inspect=None)

    # Plot image
    if colorImage:  # RGB
        imc = np.ones((im.shape[0], im.shape[1], 4), dtype=np.uint32) * 255
        imc[:, :, 0:3] = im

        img = np.empty((h, w), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((h, w, 4))
        view[:, :, :] = np.flipud(np.asarray(imc))
        p.image_rgba(image=[img], x=0, y=h, dw=w, dh=h)
    else:
        p.image(image=[np.flip(im, 0)], x=0, y=h, dw=w, dh=h, palette=palettes.Greys256)
    p.quad(top=[h], bottom=[0], left=[0], right=[w], alpha=0)  # clear rectange hack for tooltip image x,y
    p.add_tools(hoverImage)

    # Show plot
    colors = bokeh_colors(3)
    if im2 is None:
        if p1 is not None:
            p.circle(p1[:, 0], p1[:, 1], color=colors[0], line_width=2)
        if p2 is not None:
            p.circle(p2[:, 0], p2[:, 1], color=colors[1], line_width=2)
            p.multi_line(np.concatenate((p1[:, 0, None], p2[:, 0, None]), 1).tolist(),
                         np.concatenate((p1[:, 1, None], p2[:, 1, None]), 1).tolist(),
                         color='white', alpha=.3, line_width=1)

        io.show(p)  # open a browser
    else:
        h, w = im2.shape
        q = plotting.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(600 * w / h), plot_height=600,
                            x_axis_label='pixel (1 - ' + str(w) + ')', y_axis_label='pixel (1 - ' + str(h) + ')',
                            title='image', tools='box_zoom,pan,save,reset,wheel_zoom,crosshair',
                            active_scroll='wheel_zoom', active_inspect=None)
        q.add_tools(hoverImage)

        # Plot image
        q.image(image=[np.flip(im2, 0)], x=0, y=h, dw=w, dh=h, palette=palettes.Greys256)
        q.quad(top=[h], bottom=[0], left=[0], right=[w], alpha=0)  # clear rectange hack for tooltip image x,y

        # Show plots
        io.show(column(p, q))  # open a browser
