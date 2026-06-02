"""filesystemtools — separate file per tool, all standalone functions."""
from src.tools.filesystemtools.write_file import write_text, write_json, write_bytes
from src.tools.filesystemtools.read_file import read_text, read_json, read_bytes
from src.tools.filesystemtools.edit_file import append_to_file, replace_in_file, insert_at_line
from src.tools.filesystemtools.delete_file import delete_file, delete_directory, cleanup_temp_files
from src.tools.filesystemtools.list_files import list_files, list_by_extension, get_file_size, get_file_info
from src.tools.filesystemtools.make_directory import make_dir, make_temp_dir, make_reports_dir
from src.tools.filesystemtools.hash_text import hash_string, hash_file, cache_key
from src.tools.filesystemtools.truncate_text import truncate, truncate_to_tokens_approx
from src.tools.filesystemtools.save_chunk import save_chunk, load_chunk

__all__ = [
    # write
    "write_text", "write_json", "write_bytes",
    # read
    "read_text", "read_json", "read_bytes",
    # edit
    "append_to_file", "replace_in_file", "insert_at_line",
    # delete
    "delete_file", "delete_directory", "cleanup_temp_files",
    # list
    "list_files", "list_by_extension", "get_file_size", "get_file_info",
    # make_directory
    "make_dir", "make_temp_dir", "make_reports_dir",
    # hash
    "hash_string", "hash_file", "cache_key",
    # truncate
    "truncate", "truncate_to_tokens_approx",
    # save_chunk
    "save_chunk", "load_chunk",
]
