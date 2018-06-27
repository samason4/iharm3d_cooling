/*
 * hdf5_utils.h
 *
 */

#pragma once

#include <hdf5.h>

// hdf5_utils.c
hid_t hdf5_create(char *fname);
hid_t hdf5_open(char *fname);
void hdf5_close(hid_t file_id);
hid_t hdf5_make_str_type(size_t len);
void hdf5_write_scalar(const void *data, const char *name, hid_t file_id, hsize_t hdf5_type);
void hdf5_write_vector(const void *data, const char *name, hid_t file_id, size_t len, hsize_t hdf5_type);
void hdf5_write_tensor(const void *data, const char *name, hid_t file_id, size_t n1, size_t n2, hsize_t hdf5_type);
void hdf5_write_restart_prims(const void *data, const char *name, hid_t file_id);

void hdf5_write_single_val(const void *val, const char *name, hid_t file_id, hsize_t hdf5_type);

void hdf5_write_str_list(const void *data, const char *name, hid_t file_id, size_t strlen, size_t len);
void hdf5_add_units(const char *name, const char *unit, hid_t file_id);

void hdf5_make_directory(const char *name, hid_t file_id);
void hdf5_set_directory(const char *path);

void hdf5_read_single_val(void *val, const char *name, hid_t file_id, hsize_t hdf5_type);
void hdf5_read_restart_prims(void *data, const char *name, hid_t file_id);
