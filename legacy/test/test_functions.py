import numpy as np
import pandas as pd
from unittest import TestCase
from legacy.const import PROJECT_ROOT
from tmp import *


class MlTestCase(TestCase):
    def setUp(self) -> None:
        #self.config = get_config(testing=True)
        self.data = pd.read_csv(PROJECT_ROOT / "titanic/train.csv")

    def tearDown(self) -> None:
        pass
        # for path in [self.fuzz_log_file, self.fuzz_table_file, self.fuzz_matrix_index, self.fuzz_distance_matrix]:
        #     try:
        #         Path.unlink(path)
        #     except FileNotFoundError:
        #         pass

    def test_cramer_stat(self):
        assert cramer_stat(self.data['Sex'], self.data['Pclass'], method='old') == 0.1380133986221395

    def test_points_calulation(self):
        assert points_calulation(self.data['Age']) == [-np.inf, 5.0, 17.0, 19.0, 23.5, 25.0, 29.0, 31.0,
                                                       34.5, 37.0, 41.0, 46.0, 53.0, np.inf]
        self.data['AAAA'] = None
        assert points_calulation(self.data['AAAA']) == [-np.inf, np.inf]
        del self.data['AAAA']

        self.data['BBBB'] = 3
        assert points_calulation(self.data['BBBB']) == [-np.inf, np.inf]
        del self.data['BBBB']

    def test_points_calculation_tree(self):
        assert points_calculation_tree(self.data['Age'], self.data['Survived']) == [-np.inf, 6.5, 17.5, 21.5, 26.5,
                                                                                    28.75, 30.75, 33.5, 36.25, 40.25,
                                                                                    47.5, np.inf]

        self.data['AAAA'] = None
        assert points_calculation_tree(self.data['AAAA'], self.data['Survived']) == [-np.inf, np.inf]
        del self.data['AAAA']

        self.data['BBBB'] = 3
        assert points_calculation_tree(self.data['BBBB'], self.data['Survived']) == [-np.inf, np.inf]
        del self.data['BBBB']

    def test_get_thresholds_for_data(self):
        assert points_calculation_tree(self.data['Age'], self.data['Survived']) == [-np.inf, 6.5, 17.5, 21.5, 26.5,
                                                                                    28.75, 30.75, 33.5, 36.25, 40.25,
                                                                                    47.5, np.inf]
        tmp_data = self.data[['Age', 'Survived']]
        tmp_data['AAAA'] = None
        tmp_data['BBBB'] = 3
        output = {
            "Age": [-np.inf, 6.5, 17.5, 21.5, 26.5, 28.75, 30.75, 33.5, 36.25, 40.25, 47.5, np.inf],
            "AAAA": [-np.inf, np.inf],
            "BBBB": [-np.inf, np.inf],
        }
        assert output == get_thresholds_for_data(data=tmp_data, target='Survived')

    def test_var_stat(self):
        test_config = PlotConfig(
            figure_size=(30, 30),
            colormap='Blues',
            title='KEK',
        )
        var_stat(self.data, var='Age', target='Survived',  binargs={'min_size': 10}, annotation=True, plot_config=test_config)

    def test_cross_var_plot(self):
        test_config = PlotConfig(
            cbar_location='bottom'
        )
        cross_var_plot2(self.data, var1='Age', var2='Embarked', target='Survived',
                       binargs={'min_size': 10}, show_min=1, rnd=0, table=False, plot_config=None)