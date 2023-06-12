#!/usr/bin/env python
# coding: utf-8

# In[303]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import HTML

class auto_toc():
    """
    Class for Table of Contents.
    """
    def __init__(self,
                 font_size=14,
                 font_style='default',
                 sep='.',
                 header_bg='black',
                 header_color='white',
                 header_align='center'):
        """
        Initialize class variables/attributes
        """
        self.fig_num = 0
        self.table_num = 0
        self.figures = {}
        self.tables = {}
        self.font_size = font_size
        self.font_style = font_style
        self.sep = sep
        self.header_bg = header_bg
        self.header_color = header_color
        self.header_align = header_align

        # Make a folder for figures
        if not os.path.exists('figures'):
            os.makedirs('figures')


    def add_fig(self,
                title,
                name='figure',
                caption='',
                width=80,
                dpi='figure'):
        """
        Saves the current figure into an image and center-display using HTML
        """
        # Initialize variables
        folder = 'figures'
        ext = '.png'
        self.fig_num += 1
        name += str(self.fig_num)

        # Overwrites existing image
        if name[-4:] == ext:
            fp = os.path.join(folder, name)
        else:
            fp = os.path.join(folder, name+ext)
        if os.path.exists(fp):
            os.remove(fp)
        plt.savefig(fp, dpi=dpi)
        plt.close()

        # Create an HTML img tag to display the image
        img_tag = (f'<img src="{fp}" alt="plots"'
                   'style="display:block; margin-left:auto;'
                   f'margin-right:auto;width:{width}%;">') 

        # Display the img tag in the Jupyter Notebook
        display(HTML(img_tag))

        # Display the figure caption
        display(HTML(
            f"""
            <center style="font-size:{self.font_size}px;
                      font-style:{self.font_style};">
            <b>Figure {self.fig_num}{self.sep}</b> {title}.
            </center>
            <center style="font-size:{int(self.font_size*.9)}px;"><i>
            {caption}.
            </i></center>
            """
        ))

        # Save title to figures dictionary
        self.figures[self.fig_num] = title


    def add_table(self,
                  data,
                  title=None,
                  caption='',
                  index=False,
                  preview=True,
                  n_rows=10,
                  border=True,
                  header=True):
        """
        Displays the given dataframe or table data using HTML
        """
        # Initialize variables
        self.table_num += 1

        # Check data
        if isinstance(data, str):
            if data.startswith('<table>') and data.endswith('</table'):
                data = pd.read_html(data)

        if not isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            raise TypeError()

        # Auto-Formatting
        borders = [{'selector': 'tr',
                    'props': (f'border-left: 1px solid {self.header_bg};'
                              f'border-right: 1px solid {self.header_bg};'
                              f'border-bottom: 1px solid {self.header_bg};')},
                   {'selector': 'td',
                    'props': (f'border-left: 1px solid {self.header_bg};'
                              f'border-right: 1px solid {self.header_bg};'
                              f'border-bottom: 1px solid {self.header_bg};')}]
        headers = {'selector': 'th.col_heading',
                    'props': (f'background-color: {self.header_bg};'
                              f'color: {self.header_color};'
                              f'text-align: {self.header_align}')}

        styles = [headers]
        if border:
            styles.extend(borders)

        if index:
            data.reset_index(inplace=True)

        prefix = ''
        if preview:
            data = data.head(n_rows)
            prefix = 'Preview of '

        # Display the table
        data = data.style.set_table_styles(styles).hide()
        display(HTML(f'<center>{data.to_html()}</center>'))

        # Display the table caption
        display(HTML(
            f"""
            <center style="font-size:{self.font_size}px;
                           font-style:{self.font_style};">
            <b>Table {self.table_num}{self.sep}</b> {prefix}{title}.
            </center>
            <center style="font-size:{int(self.font_size*.9)}px;"><i>
            {caption}.
            </i></center>
            """
        ))

        # Save title to figures dictionary
        self.tables[self.table_num] = title


    def print_toc(self):
        """Prints the markdown values for the table of contents cell"""
        print_out = '# Table of Contents'
    
        if len(self.tables.keys()) > 0:
            print_out += '\n**Tables**'
            for number, table in self.tables.items():
                print_out += f'\n* Table {number}. {table}'
            print_out += '\n\n'
            
        if len(self.figures.keys()) > 0:
            print_out += '\n**Figures**'
            for number, figure in self.figures.items():
                print_out += f'\n* Figure {number}. {figure}'
                
        print(print_out)
