#!/usr/bin/env python3

from wand.image import Image
from wand.display import display
from wand.color import Color

# Images:
# Baseline images (seconds apart)
seconds_baseline_1 = '../court_two/no activity/2020-01-15_08.09.45.jpeg'
seconds_1 = '../court_two/no activity/2020-01-15_08.09.45.jpeg'
seconds_baseline_2 = '../court_two/no activity/2020-01-15_08.19.05.jpeg'
seconds_2 = '../court_two/no activity/2020-01-15_08.19.36.jpeg'
seconds_baseline_3 = '../court_two/no activity/2020-01-15_11.28.28.jpeg'
seconds_3 = '../court_two/no activity/2020-01-15_11.29.09.jpeg'

# Baseline pairs
seconds_1 = ('\n(Lower is better) seconds_1', seconds_baseline_1, seconds_1)
seconds_2 = ('(Lower is better) seconds_2', seconds_baseline_2, seconds_2)
seconds_3 = ('(Lower is better) seconds_3', seconds_baseline_3, seconds_3)


# Drop tuples
camera_drop_baseline_1 = './low activity/2020-01-15_11.49.40.jpeg'
camera_drop_1 = './low activity/2020-01-15_07.25.08.jpeg'

# Drop pairs
camera_drop_1 = ('\n(Lower is better) camera_drop_1', camera_drop_1, camera_drop_baseline_1)


# Shadow images
shadow_baseline_1 = './no activity/2020-01-15_10.52.53.jpeg'
shadow_1 = './no activity/2020-01-15_10.54.02.jpeg'
shadow_baseline_2 = './no activity/2020-01-15_14.58.41.jpeg'
shadow_2 = './no activity/2020-01-15_14.29.49.jpeg'

# Shadow pairs
shadow_1 = ('(Lower is better) shadow_1', shadow_baseline_1, shadow_1)
shadow_2 = ('(Lower is better) shadow_2', shadow_baseline_2, shadow_2)


# Object appears images
object_baseline_1 = './low activity/2020-01-15_16.51.31.jpeg'
object_1 = './low activity/2020-01-15_16.48.36.jpeg'
object_baseline_2 = './low activity/2020-01-15_16.53.19.jpeg'
object_2 = './low activity/2020-01-15_17.04.53.jpeg'
object_baseline_3 = './low activity/2020-01-15_16.51.31.jpeg'
object_3 = './low activity/2020-01-15_17.31.02.jpeg'

# Object appears pairs
object_1 = ('\n(Higher is better) object_1', object_baseline_1, object_1)
object_2 = ('(Higher is better) object_2', object_baseline_2, object_2)
object_3 = ('(Higher is better) object_3', object_baseline_3, object_3)

# High activity images
activity_baseline_1 = '../dog_park/high activity/2020-01-15_17.07.37.jpeg'
activity_1 = '../dog_park/high activity/2020-01-15_17.09.40.jpeg'
activity_baseline_2 = activity_baseline_1
activity_2 = '../dog_park/high activity/2020-01-15_16.43.04.jpeg'

# High activity pairs
activity_1 = ('\n(Higher is better) activity_1', activity_baseline_1, activity_1)
activity_2 = ('(Higher is better) activity_2', activity_baseline_2, activity_2)

comparisons = [
    seconds_1,
    seconds_2,
    seconds_3,
    camera_drop_1,
    shadow_1,
    shadow_2,
    object_1,
    object_2,
    object_3,
    activity_1,
    activity_2,
]

comparison_metrics = [
    'undefined',
    'absolute',
    'fuzz',
    'mean_absolute',
    'mean_error_per_pixel',
    'mean_squared',
    'normalized_cross_correlation',
    'peak_absolute',
    'peak_signal_to_noise_ratio',
    'perceptual_hash',
    'root_mean_square',
    'structural_similarity',
    'structural_dissimilarity',
]


for comparison_metric in comparison_metrics:
    print(comparison_metric + ':')
    for comparison_label, img1, img2 in comparisons:
        with Image(filename=img1) as base:
            with Image(filename=img2) as change:
                _, result_metric = base.compare(change, metric=comparison_metric, highlight=Color('white'), lowlight=Color('black'))
                diff_percent = round(result_metric * 100, 4)

                print('  ' + comparison_label + ': ' + str(diff_percent) + ' %')

    input('Press Enter to proceed...')
    print('\n')