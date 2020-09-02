split_objects = {
  'train': {
    'exclude_objects': ['mug', 'pan', 'wine_glass'],
  },
  'test': {
    'include_objects': ['mug', 'pan', 'wine_glass']
  }
}

split_participants = {
  'train': {
    'exclude_sessions': [5, 15, 25, 35, 45],
  },
  'test': {
    'include_sessions': [5, 15, 25, 35, 45],
  }
}

split_overfit = {
  'train': {
    'include_sessions': [50],
    'include_instructions': ['use'],
    'include_objects': ['mug', 'pan', 'wine_glass']
  },
  'test': {
    'include_sessions': [50],
    'include_instructions': ['use'],
    'include_objects': ['mug', 'pan', 'wine_glass']
  }
}

split_images_objects = {
  'train': {
    'include_sessions': list(range(19, 25)) + list(range(26, 30)) + list(range(39, 45)) + list(range(46, 50)),
    'exclude_objects': ['mug', 'pan', 'wine_glass'],
  },
  'test': {
    'include_objects': ['mug', 'pan', 'wine_glass'],
  }
}

split_images_participants = {
  'train': {
    'include_sessions': list(range(19, 25)) + list(range(26, 30)) + list(range(39, 45)) + list(range(46, 50)),
  },
  'test': {
    'include_sessions': [5, 15, 25, 35, 45],
  }
}
