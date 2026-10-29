"""Global-motion analysis of the r9 run: population-mean trajectory FFT,
amplitude at 5.69 Hz vs 7.11 Hz, px conversion via COLMAP geometry."""
import numpy as np, json, struct

FPS = 20.003929910241613
d = np.load('runs/lego_turbine_2_missing_r9-415299ad/trajectories.npz')
traj = d['traj'].astype(np.float64); xyz = d['canonical_xyz'].astype(np.float64)
op = d['opacity'].astype(np.float64); times = d['times'].astype(np.float64)
keep = op > 0.1
traj, xyz = traj[keep], xyz[keep]
T = traj.shape[1]
duration = 164 / FPS

disp = traj - xyz[:, None, :]
mean_disp = disp.mean(axis=0)              # (T,3) global mean displacement
mean_disp_c = mean_disp - mean_disp.mean(axis=0)
fft = np.fft.rfft(mean_disp_c, axis=0)
freqs = np.fft.rfftfreq(T, d=(times[1]-times[0])*duration)
amp = np.abs(fft)*2.0/T
amp_tot = np.linalg.norm(amp, axis=1)

top = sorted([(round(float(freqs[i]),2), round(float(amp_tot[i]),5)) for i in range(1,len(freqs))], key=lambda x:-x[1])[:8]
i569 = int(np.argmin(np.abs(freqs-5.69))); i711 = int(np.argmin(np.abs(freqs-7.112)))
print('global mean trajectory spectrum top8:', top)
print(f'amp@5.69Hz bin({freqs[i569]:.2f}): {amp_tot[i569]:.5f}')
print(f'amp@7.11Hz bin({freqs[i711]:.2f}): {amp_tot[i711]:.5f}')
print(f'global mean max disp: {np.linalg.norm(mean_disp,axis=1).max():.4f} scene units')

# per-point amp at 5.69 Hz
disp_c = disp - disp.mean(axis=1, keepdims=True)
fft_p = np.fft.rfft(disp_c, axis=1)
amp_p = np.linalg.norm(np.abs(fft_p)*2.0/T, axis=-1)
print(f'per-point amp@5.69Hz: median {np.median(amp_p[:,i569]):.5f}, p90 {np.percentile(amp_p[:,i569],90):.5f}')
proj = disp_c @ (np.linalg.svd(mean_disp_c, full_matrices=False)[2][0])
ph = np.angle(np.fft.rfft(proj, axis=1)[:, i569])
strong = amp_p[:,i569] > np.percentile(amp_p[:,i569], 50)
R = np.abs(np.exp(1j*ph[strong]).mean())
print(f'phase coherence @5.69Hz (strong half): R={R:.3f}')

# px conversion: median camera distance from COLMAP sparse
def read_next(fid,n,fmt,e='<'):
    b=fid.read(n); return b, struct.unpack(e+fmt,b)
cams=[]
with open('data/multipleview/lego_turbine_2_missing/sparse_/images.bin','rb') as f:
    _,(n,)=read_next(f,8,'Q')
    for _ in range(n):
        _,vals=read_next(f,64,'idddddddi')
        q=np.array(vals[1:5]); t=np.array(vals[5:8])
        name=b''
        while True:
            c=fid.read(1)
            if c==b'\x00': break
            name+=c
        _,(n2,)=read_next(f,8,'Q')
        f.read(24*n2)
        w,x,y,z=q
        Rm=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
        C=-Rm.T@t
        cams.append(C)
cams=np.array(cams)
center = xyz.mean(axis=0)
dist = np.linalg.norm(cams-center, axis=1)
print(f'camera distances to cloud center: median {np.median(dist):.1f} scene units')
f_px = 1846.1
px_per_unit = f_px/np.median(dist)
print(f'px per scene unit at center depth: {px_per_unit:.1f}')
print(f'global mean max disp in px (full-res): {np.linalg.norm(mean_disp,axis=1).max()*px_per_unit:.1f}')
print(f'per-point p90 amp@5.69 in px: {np.percentile(amp_p[:,i569],90)*px_per_unit:.2f}')
print(f'median max-disp in px: {np.median(np.linalg.norm(disp,axis=-1).max(axis=-1))*px_per_unit:.1f}')
