
print('real vel: ', np.linalg.norm(np.array(kerbin_body.velocity(inertial_frame_sun))))

kerb_vel = np.array(kerbin_body.velocity(inertial_frame_sun))
ksp_vec_to_normal(kerb_vel)
print(np.array(start_vel) * 1000, kerb_vel, )
print(
    'вылетная скорость без инерциальной:',
    np.array(start_vel) * 1000 + kerb_vel,
    np.linalg.norm(np.array(start_vel) * 1000 + kerb_vel)
)

print(np.linalg.norm(start_vel_u))
print(tuple(np.array(start_vel_u) * soi_radius))
print(start_vel * 1000)

pos = np.array(start_vel_u) * soi_radius
ksp_vec_to_normal(pos)

vel = start_vel * 1000
ksp_vec_to_normal(vel)

v = conn.space_center.transform_velocity(
    tuple(pos),
    tuple(vel),
    inertial_frame_kerbin,
    inertial_frame_sun
)