import copy


class PlotConfig:
    def __init__(
            self,
            style='seaborn-ticks',
            figure_size=(20, 20),
            colormap=None,
            annotation_font_size=15,
            line_widths=1,
            cbar_location='right',
            cbar_width=5,
            cbar_pad=1,
            cbar_tick_loc=None,
            title=None,
            title_size=22,
            title_pad=1,
            xtick_loc=None,
            ytick_loc=None,
            y_inverse=False,
            x_inverse=False,
            xlabel_size=17,
            ylabel_size=17,
            xtick_size=17,
            ytick_size=17,
            y_rotation=None,
            x_rotation=None,
            xlabel=None,
            ylabel=None,
            y2label=None,
            annotation_delta=None,
            color=None,
            color2=None,
            y_grid: bool = None,
            y2_grid: bool = None,
            x_grid: bool = None,
            y_color=None,
            y2_color=None,
            ymax=None,
            ymin=None,
            y2min=None,
            y2max=None,
            width_ratios=None,
            height_ratios=None,
            wspace=None,
            bar_width=None,
            side_grid=None
    ):
        """

        :param style: Style of plot, available values in constant POSSIBLE_STYLES
        :param figure_size: Size of plot
        :param colormap: Colormap of plot
        :param annotation_font_size: Font size of annotation text on plot
        :param line_widths: For some types of plots it is a with of grid, for other it is width of lines
        :param cbar_location: Colorbar location - left/right/bottom/top
        :param cbar_width: Width of Colorbar
        :param cbar_pad: Pad of Colorbar
        :param cbar_tick_loc: Tickets location on Colorbar
        :param title: Title of the plot
        :param title_size: Font size of title
        :param title_pad: Pad of title
        :param xtick_loc: Position of X axis - bottom/top
        :param ytick_loc: Position of Y axis - right/left
        :param y_inverse: Inversion of Y axis
        :param x_inverse: Inversion of X axis
        :param xlabel_size: Font size of X-Label
        :param ylabel_size: Font size of Y-Label
        :param xtick_size: Font size of X-tickets
        :param ytick_size: Font size of Y-tickets
        :param y_rotation: Rotation of Y-tickets
        :param x_rotation: Rotation of X-tickets
        :param xlabel: Label of X-axis
        :param ylabel: Label of Y-axis
        :param y2label: Label of Y2-axis
        :param annotation_delta: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        :param color: Color of bars, lines
        :param color2: Spare Color for bars, lines
        :param y_grid: Show Y-grid
        :param x_grid: Show X-grid
        :param y2_grid: Show Y2-grid
        :param y_color: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        :param y2_color: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        :param ymax: Max value on Y-axis
        :param ymin: Min value on Y-axis
        :param y2min: Min value on Y2-axis
        :param y2max: Max value on Y2-axis
        """
        self.style = style
        self.figure_size = figure_size
        self.colormap = colormap
        self.annotation_font_size = annotation_font_size
        self.line_widths = line_widths
        self.cbar_location = cbar_location
        self.cbar_width = cbar_width
        self.cbar_pad = cbar_pad
        self.cbar_tick_loc = cbar_tick_loc
        self.title = title
        self.title_size = title_size
        self.title_pad = title_pad
        self.xtick_loc = xtick_loc
        self.ytick_loc = ytick_loc
        self.y_inverse = y_inverse
        self.x_inverse = x_inverse
        self.xlabel_size = xlabel_size
        self.ylabel_size = ylabel_size
        self.xtick_size = xtick_size
        self.ytick_size = ytick_size
        self.y_rotation = y_rotation
        self.x_rotation = x_rotation
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.y2label = y2label
        self.annotation_delta = annotation_delta
        self.color = color
        self.color2 = color2
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.y2_grid = y2_grid
        self.y_color = y_color
        self.y2_color = y2_color
        self.ymax = ymax
        self.ymin = ymin
        self.y2min = y2min
        self.y2max = y2max

        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.wspace = wspace
        self.bar_width = bar_width
        self.side_grid = side_grid

    def merge_other_config(self, config):
        all_attr = self.__dict__.keys()
        all_params = dict()
        for attr in all_attr:
            v1 = self.__getattribute__(attr)
            v2 = config.__getattribute__(attr)
            if v2 is not None and v2 != v1:
                all_params[attr] = v2
            else:
                all_params[attr] = v1

        return PlotConfig(**all_params)

    def appy_configs(self, fig, ax=None, ax2=None, cax=None, side_axis=None):
        if ax:
            if self.title:
                if side_axis:
                    fig.suptitle(self.title, fontsize=self.title_size, y=0.98 + self.title_pad)
                else:
                    ax.set_title(self.title, fontdict={'fontsize': self.title_size}, y=self.title_pad)
            if self.xtick_loc == 'top':
                ax.xaxis.tick_top()
            if self.ytick_loc == 'right':
                ax.yaxis.tick_right()
            if self.y_inverse:
                ax.invert_yaxis()
            if self.x_inverse:
                ax.invert_xaxis()

            ax.xaxis.label.set_fontsize(self.xlabel_size)
            ax.yaxis.label.set_fontsize(self.ylabel_size)

            for item in ax.get_xticklabels():
                item.set_fontsize(self.xtick_size)
            for item in ax.get_yticklabels():
                item.set_fontsize(self.ytick_size)
                if self.y_color is not None:
                    item.set_color(self.y_color)

            if self.y_rotation is not None:
                ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=self.y_rotation)
            if self.x_rotation is not None:
                ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=self.x_rotation)

            if self.x_grid is not None:
                ax.xaxis.grid(self.x_grid)
            if self.y_grid is not None:
                ax.yaxis.grid(self.y_grid)

            if self.xlabel is not None:
                ax.set_xlabel(self.xlabel)

            if self.ylabel is not None:
                ax.set_ylabel(self.ylabel)

            if self.y_color is not None:
                ax.yaxis.label.set_color(self.y_color)

            if self.ymax is not None:
                ax.set_ylim(top=self.ymax)
            if self.ymin is not None:
                ax.set_ylim(bottom=self.ymin)

        if cax:
            if self.cbar_tick_loc == 'right':
                cax.yaxis.tick_right()
            elif self.cbar_tick_loc == 'left':
                cax.yaxis.tick_left()
            elif self.cbar_tick_loc == 'top':
                cax.xaxis.tick_top()
            elif self.cbar_tick_loc == 'bottom':
                cax.xaxis.tick_bottom()

        if ax2:
            if self.y2_grid is not None:
                ax2.yaxis.grid(self.y2_grid)
            if self.y2label is not None:
                ax2.set_ylabel(self.y2label)

            ax2.yaxis.label.set_fontsize(self.ylabel_size)

            if self.y2_color is not None:
                ax2.yaxis.label.set_color(self.y2_color)

            for item in ax2.get_yticklabels():
                item.set_fontsize(self.ytick_size)
                if self.y2_color is not None:
                    item.set_color(self.y2_color)

            if self.y2max is not None:
                ax2.set_ylim(top=self.y2max)
            if self.y2min is not None:
                ax2.set_ylim(bottom=self.y2min)

        if side_axis:
            for sax in side_axis:
                if self.side_grid is not None:
                    sax.xaxis.grid(self.side_grid)
                    sax.yaxis.grid(self.side_grid)

    def __copy__(self):
        return PlotConfig(**self.__dict__)


BASE_CONFIG = PlotConfig()
