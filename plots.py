def plotresults(cam, im, P, S, bbox):
    from bokeh import io, plotting, palettes, models
    from bokeh.layouts import column

    # Bokeh Plotting
    io.reset_output()
    io.output_file('bokeh plots.html', title='bokeh plots')
    h, w = im.shape
    n = P.shape[2]
    xn = list(range(0, n, 1))
    colors = bokeh_colors(palettes, n)

    x = P[0,]  # x
    y = P[1,]  # y

    # Setup tools
    hoverImage = models.HoverTool(tooltips=[('index', '$index'), ('(x, y)', '($x{(1.11)}, $y{(1.11)})')],
                                  point_policy='follow_mouse')

    hover = models.HoverTool(tooltips=[('index', '$index'), ('(x, y)', '(@x{(1.11)}, @y{(1.11)})')],
                             point_policy='snap_to_data', mode='vline')  # $x refers to mouse pos, @x to datapoint pos

    # Setup widgets
    # slider = bokeh.models.widgets.Slider(start=1, end=9, value=1, step=1, title="Image")

    # Set up callbacks
    # def update_title(attrname, old, new):
    #     p.title.text = 'changed'  # str(slider.value)
    #     print('ran')
    # slider.on_change('value', update_title)

    p = plotting.figure(x_range=(0, w), y_range=(h, 0), plot_width=round(800 * w / h), plot_height=800,
                        title=cam['filename'],
                        x_axis_label='pixel (1 - ' + str(w) + ')',
                        y_axis_label='pixel (1 - ' + str(h) + ')',
                        tools='box_zoom,pan,save,reset,wheel_zoom,crosshair',
                        active_scroll='wheel_zoom',
                        active_inspect=None)
    p.add_tools(hoverImage)

    # Plot image
    p.image(image=[im], x=0, y=h, dw=w, dh=h, palette=palettes.Greys256)

    # Plot clear rectangle
    p.quad(top=[h], bottom=[0], left=[0], right=[w], alpha=0)  # clear rectange hack for tooltip image x,y

    # Plog bounding box
    p.quad(top=bbox[1] + bbox[3], bottom=bbox[1], left=bbox[0], right=bbox[0] + bbox[2], color=colors[1], line_width=2,
           alpha=.3)  # clear rectange hack for tooltip image x,y

    # Plot license plate outline
    p.patch(x[0:4, 0], y[0:4, 0], alpha=0.3, line_width=4)

    # Plot points
    p.circle(P[2,].ravel(), P[3,].ravel(), color=colors[1], legend='Reprojections', size=9, alpha=.7)
    p.circle(P[0,].ravel(), P[1,].ravel(), color=colors[0], legend='Points', line_width=2)

    # Plot points (color by frame)
    # for i in np.arange(P.shape[2]):
    #     p.circle(x[:, i], y[:, i], color=colors[i], legend='image ' + str(i), line_width=2)
    #     p.circle(P[3, :, i], P[4, :, i], color=colors[i], legend='image ' + str(i), size=12, alpha=.3)

    # Plot lines
    p.multi_line(x.tolist(), y.tolist(), color='white', alpha=.8, line_width=1)

    # Legend
    p.legend.click_policy = 'hide'

    # Second Plot
    q = plotting.figure(x_range=(0, n), y_range=(0, 60), plot_width=300, plot_height=300,
                        title='Speed',
                        x_axis_label='image',
                        y_axis_label='speed (km/h)',
                        tools='save,reset,wheel_zoom',
                        active_scroll='wheel_zoom',
                        active_inspect=hover)
    q.circle(xn[1:], S[1:, 8], color=colors[0], line_width=2)
    q.line(xn[1:], S[1:, 8], color=colors[0], line_width=2)
    q.add_tools(hover)

    # Show plot
    # widgets = widgetbox(slider)
    # io.show(column(p, widgets))  # open a browser
    # io.show(p)  # open a browser
    io.show(column(p, q))  # open a browser


def bokeh_colors(palettes, n):
    # https: // bokeh.pydata.org / en / latest / docs / reference / palettes.html
    # returns appropriate 10, 20 or 256 colors for plotting. n is the maximum required colors
    if n < 11:
        return palettes.Category10[10]
    elif n < 21:
        return palettes.Category20[20]
    else:
        return palettes.Viridis256
