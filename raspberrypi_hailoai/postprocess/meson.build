
################################################
# river_trash_post SOURCES
################################################

river_trash_post_sources = [
    'river_trash_post.cpp',
]
shared_library('river_trash_post',
    river_trash_post_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: join_paths(meson.project_source_root(), 'resources'),
)