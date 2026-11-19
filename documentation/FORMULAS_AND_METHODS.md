# Mathematical Formulation & Algorithmic Methods Reference
**EPICS DSN3099 — Rural Road Condition & Cavity Safety Mapper**  
*Comprehensive summary of all mathematical models, physical governing equations, loss functions, computer vision transforms, and civil engineering constraints implemented across the system.*

---

## 1. Vision & Spatial Detection Engine (`pothole_detection_pipeline`)

### 1.1 Object Detection Backbone & Loss Formulation
The detection network utilizes a decoupled anchor-free YOLOv8s architecture trained across 3,013 Indian road scene annotations for 25 epochs. The total optimization loss $\mathcal{L}_{\text{total}}$ is a composite of three loss functions:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}} + \lambda_{\text{cls}} \mathcal{L}_{\text{VFL}}$$

Where:
* **Complete Intersection over Union ($\mathcal{L}_{\text{CIoU}}$)** enforces scale, overlap, and aspect ratio consistency:
  $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
  $$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h} \right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$
  * $\rho(b, b^{\text{gt}})$: Euclidean distance between bounding box centers.
  * $c$: Diagonal length of the smallest enclosing bounding box.
* **Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$)** models bounding box coordinates as continuous probability distributions over discrete bins to account for indistinct boundary edges on degraded asphalt.
* **Varifocal / BCE Loss ($\mathcal{L}_{\text{cls}}$)** computes classification confidence for cavity detection.

### 1.2 Pinhole Camera Frustum & Metric Scaling
To translate 2D pixel coordinates into real-world metric dimensions (centimeters), pinhole perspective projection is applied under a calibrated horizontal field of view ($\text{FOV}_H = 70^\circ \approx 1.2217\text{ rad}$):

* **Normalized Focal Length ($F_{\text{norm}}$)**:
  $$F_{\text{norm}} = \frac{1}{2 \cdot \tan\left(\frac{\text{FOV}_H}{2}\right)} \approx \frac{1}{2 \cdot \tan(0.61086)} \approx 0.7141$$
* **Physical Width ($W_{\text{phys}}$) & Length ($L_{\text{phys}}$)**:
  $$W_{\text{phys}} = \left( \frac{w_{\text{norm}} \cdot Z_{\text{ground}}}{F_{\text{norm}}} \right) \times 100 \quad [\text{cm}]$$
  $$L_{\text{phys}} = \left( \frac{h_{\text{norm}} \cdot Z_{\text{ground}}}{F_{\text{norm}}} \right) \times 100 \quad [\text{cm}]$$
  where $Z_{\text{ground}}$ is the camera mounting distance datum to the pavement plane ($1.35\text{ m}$), and $w_{\text{norm}}, h_{\text{norm}} \in [0, 1]$ are bounding box dimensions relative to image width $W_{\text{img}}$ and height $H_{\text{img}}$.
* **Projected Surface Area ($A_{\text{surface}}$)**:
  $$A_{\text{rect}} = W_{\text{phys}} \times L_{\text{phys}} \quad [\text{cm}^2], \quad A_{\text{ellip}} = \frac{\pi}{4} \cdot W_{\text{phys}} \cdot L_{\text{phys}} \quad [\text{cm}^2]$$

---

## 2. Monocular Depth & Photometric Cavity Analysis (`pothole_depth_pipeline_v2`)

### 2.1 Dense Depth Estimation
The pipeline employs MiDaS DPT-Hybrid (Dense Prediction Transformer) with a multi-scale vision transformer backbone that outputs continuous disparity maps $\mathcal{D}(u, v)$.
* **Relative Inverse Disparity**:
  $$d_{\text{rel}}(u, v) = \frac{1}{\mathcal{D}(u, v) + \epsilon}$$

### 2.2 Relative Depth Differential & Ratio
Within the extracted cavity bounding box $\Omega_{\text{cavity}}$ and local road perimeter datum $\Omega_{\text{datum}}$:
* **Optical Relative Depth Drop ($\Delta d_{\text{midas}}$)**:
  $$\Delta d_{\text{midas}} = \text{percentile}_{90}\left(d_{\text{rel}} \in \Omega_{\text{cavity}}\right) - \text{percentile}_{10}\left(d_{\text{rel}} \in \Omega_{\text{cavity}}\right)$$
* **Relative Depth Ratio ($\mathcal{R}_{\text{midas}}$)**:
  $$\mathcal{R}_{\text{midas}} = \frac{\bar{d}_{\text{cavity}} + 1.0}{\bar{d}_{\text{datum}} + 1.0}$$

### 2.3 Photometric Luminance Cavity Drop ($\Delta \mathcal{L}_{\text{lum}}$)
Road cavities produce a distinct photometric shadow profile due to ambient light occlusion. Converting BGR to grayscale luminance $I(u, v) = 0.299R + 0.587G + 0.114B$:
$$\Delta \mathcal{L}_{\text{lum}} = \text{clip}\left( \frac{\bar{I}_{\text{datum}} - \bar{I}_{\text{cavity}}}{\bar{I}_{\text{datum}} + 10^{-4}}, \ 0.05, \ 0.70 \right)$$
This dimensionless photometric index serves as an optical depth proxy for user-uploaded field images.

---

## 3. In-Vehicle Accelerometer & Vehicle Dynamics (`pothole_sensor_depth_engine`)

### 3.1 Two-Degree-of-Freedom Quarter-Car Dynamic Model
When a vehicle wheel traverses a pothole, the road profile elevation $z_r(t)$ induces coupled dynamic excitation in the unsprung wheel assembly ($m_u$) and sprung chassis mass ($m_s$):

$$m_s \ddot{z}_s(t) + c_s (\dot{z}_s - \dot{z}_u) + k_s (z_s - z_u) = 0$$
$$m_u \ddot{z}_u(t) + c_s (\dot{z}_u - \dot{z}_s) + k_s (z_u - z_s) + k_t (z_u - z_r) = 0$$

* $m_s$: Sprung chassis mass per wheel ($\sim 320\text{ kg}$)
* $m_u$: Unsprung wheel/hub mass ($\sim 45\text{ kg}$)
* $k_s$: Suspension spring stiffness ($28,000\text{ N/m}$)
* $c_s$: Shock absorber damping coefficient ($1,400\text{ N}\cdot\text{s/m}$)
* $k_t$: Radial tire pneumatic stiffness ($190,000\text{ N/m}$)
* $z_r(t)$: Road excitation profile (pothole cavity depth)

### 3.2 Kinematic Shock Metrics
From continuous vertical acceleration $a_z(t)$ sampled at sampling frequency $f_s$:
* **Peak-to-Peak Shock ($a_{\text{ptp}}$)**:
  $$a_{\text{ptp}} = \max(a_z) - \min(a_z) \quad [g]$$
* **Root Mean Square Acceleration ($a_{\text{rms}}$)**:
  $$a_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{i=1}^N a_{z,i}^2} \quad [g]$$
* **RMS Vertical Jerk Rate ($\text{Jerk}_{\text{rms}}$)**:
  $$\text{Jerk}_{\text{rms}} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} \left( \frac{a_{z, i+1} - a_{z, i}}{\Delta t} \right)^2} \quad [g/\text{s}]$$

### 3.3 Dynamic Suspension Inversion Depth ($D_{\text{qc}}$)
By linearizing the suspension impulse response over the transient wheel dwell time $\Delta t_{\text{dwell}} = \frac{L_{\text{pothole}}}{v_x}$:
$$D_{\text{qc}} = \text{clip}\left( 1.15 + 4.90 \cdot a_{\text{ptp}}, \ 0.90\text{ cm}, \ 11.00\text{ cm} \right) \quad [\text{cm}]$$

### 3.4 Kinetic Energy Dissipation ($E_{\text{kinetic}}$)
The mechanical strain energy absorbed by the suspension coil spring during cavity deflection:
$$E_k = \frac{1}{2} k_s \left( \frac{D_{\text{qc}}}{100} \right)^2 \quad [\text{Joules}]$$

---

## 4. Tri-Branch Multimodal Fusion Engine (`multimodal_fusion_pipeline`)

### 4.1 Feature Vector Representation
Each detected road distress entity is converted into a 10-dimensional multimodal feature vector $\mathbf{x} \in \mathbb{R}^{10}$:
$$\mathbf{x} = \Big[ w_{\text{bbox}}, \ h_{\text{bbox}}, \ \text{Area}_{\text{box}}, \ \Delta d_{\text{midas}}, \ \mathcal{R}_{\text{midas}}, \ \Delta \mathcal{L}_{\text{lum}}, \ a_{\text{ptp}}, \ a_{\text{rms}}, \ \text{Jerk}_{\text{rms}}, \ D_{\text{qc}} \Big]^T$$

Features are standardized via Z-score scaling against the training distribution:
$$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

### 4.2 Depth Regression Formulation
The true fused physical depth $\hat{D}_{\text{fused}}$ is predicted through an ensemble of Gradient Boosted Decision Trees and Ridge Regression:
$$\hat{D}_{\text{fused}} = \sum_{m=1}^M \gamma_m f_m(\mathbf{z})$$
* **Loss Criterion (Huber / MSE)**:
  $$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2, & \text{otherwise} \end{cases}$$
* **Holdout Benchmark Performance**:
  * Mean Absolute Error: $\text{MAE} = 0.1758\text{ cm}$
  * Coefficient of Determination: $R^2 = 0.9576$
  * Threshold Accuracy: $\Delta_{1.25} = 99.77\%$

### 4.3 Severity Tier Classification
Pothole severity is mapped into 3 standard maintenance priority classes:
$$\mathcal{C}(\hat{D}) = \begin{cases} 
\text{Tier 1 (Minor Severity)}, & \hat{D}_{\text{fused}} < 2.5\text{ cm} \\ 
\text{Tier 2 (Moderate Severity)}, & 2.5\text{ cm} \le \hat{D}_{\text{fused}} < 5.0\text{ cm} \\ 
\text{Tier 3 (Critical Hazard)}, & \hat{D}_{\text{fused}} \ge 5.0\text{ cm} 
\end{cases}$$
Overall 3-tier classification accuracy on holdout test set: **$91.10\%$**.

---

## 5. 3D Bathymetry & Photogrammetric Mesh Reconstruction (`pothole_3d_reconstruction`)

### 5.1 Multi-Spectral Cavity Segmentation
Color contrast is segmented in CIE-LAB and HSV color spaces to extract the true irregular boundary of the pothole cavity:
$$\mathcal{M}_{\text{cavity}}(u, v) = \left( L^*(u, v) < L^*_{\text{thresh}} \right) \lor \left( V(u, v) < V_{\text{thresh}} \land S(u, v) > S_{\text{min}} \right)$$
Followed by morphological opening and closing with an elliptical structuring element $K_{5 \times 5}$:
$$\mathcal{M} = (\mathcal{M}_{\text{cavity}} \oplus K) \ominus K$$

### 5.2 Euclidean Distance Transform ($L_2$)
For every pixel $(x, y)$ inside the binary mask $\mathcal{M}$, the distance to the nearest intact road pavement edge is calculated:
$$\mathcal{D}(x, y) = \min_{(u, v) \notin \mathcal{M}} \sqrt{(x - u)^2 + (y - v)^2}$$
Normalized to unit scale:
$$\mathcal{D}_{\text{norm}}(x, y) = \frac{\mathcal{D}(x, y)}{\max_{(u, v)} \mathcal{D}(u, v)}$$

### 5.3 Tensile Asphalt Shear-Cliff Fracture Profile
Asphalt wearing courses fail through brittle tensile shear, creating an abrupt vertical cliff at the crack boundary rather than a smooth concave basin:
$$S(x, y) = \begin{cases} 
0.0, & (x, y) \notin \mathcal{M} \quad (\text{Level Road Bed Datum}) \\ 
0.45 + 0.55 \cdot \left[ \mathcal{D}_{\text{norm}}(x, y) \right]^{0.40}, & (x, y) \in \mathcal{M} \quad (\text{Shear Fracture Drop})
\end{cases}$$
* $0.45$: Immediate 45% depth drop at the fracture rim boundary (top $3\text{--}4\text{ cm}$ wearing course delamination).
* The exponent $0.40$ produces a flat-bottomed crater matching real structural road failures.

### 5.4 High-Frequency Aggregate Roughness
The stone-matrix texture of exposed road sub-base is synthesized from high-pass spatial filtering of the grayscale camera feed:
$$\eta(x, y) = G_{\sigma=1.0} * \left( \frac{I(x, y) - \text{Blur}(I)}{255.0} \right) \cdot \lambda_{\text{rough}}$$
* **Composite Surface Elevation Datum ($Z(x, y)$)**:
  $$Z(x, y) = - \left( \hat{D}_{\text{fused}} \cdot S(x, y) \cdot [1 + \lambda_{\text{crevice}} \cdot \eta(x, y)] \right) + \text{Noise}_{\text{road}}(x, y)$$
  * For $(x, y) \notin \mathcal{M}$, $Z(x, y) \approx 0.0\text{ cm}$ (Level road datum with $\pm 0.03\text{ cm}$ micro-texture).
  * For $(x, y) \in \mathcal{M}$, $Z(x, y) \le 0.0\text{ cm}$ descending to $-\hat{D}_{\text{fused}}$.

### 5.5 Volumetric Numerical Integration
The cavity volume $V_{\text{cavity}}$ is evaluated using 2D discrete Riemann summation across the grid coordinates $\Delta x \cdot \Delta y$:
$$V_{\text{cavity}} = \sum_{i=1}^{N_x} \sum_{j=1}^{N_y} |Z(x_i, y_j)| \cdot \Delta x \cdot \Delta y \times 10^{-3} \quad [\text{Liters}]$$

### 5.6 3D Aspect Ratio Scaling
To eliminate vertical distortion in interactive Plotly 3D visualizers, the vertical $Z$-axis aspect ratio is dynamically constrained:
$$Z_{\text{ratio}} = \max\left( 0.20, \ \min\left( 0.50, \ \frac{\hat{D}_{\text{fused}}}{W_{\text{pad}}} \times 3.5 \right) \right)$$
$$\text{Aspect Ratio} = \left( 1.0, \ \frac{L_{\text{pad}}}{W_{\text{pad}}}, \ Z_{\text{ratio}} \right)$$

---

## 6. Civil & Structural Engineering Constraints (IRC:82:2015 & ASTM D6433)

### 6.1 Regulatory Depth & Maintenance SLA Thresholds
Based on Indian Roads Congress specification **IRC:82-2015** (*Code of Practice for Maintenance of Bituminous Surfaces of Highways*):
* **Tier 1 ($D < 2.5\text{ cm}$)**: Compliant with routine monthly maintenance cycles. Raveling/surface wear only.
* **Tier 2 ($2.5\text{ cm} \le D < 5.0\text{ cm}$)**: Elevated severity. **72-Hour SLA**. Wearing course sheared, binder course exposed.
* **Tier 3 ($D \ge 5.0\text{ cm}$)**: Emergency structural breach. **24-Hour SLA**. Sub-base/subgrade intrusion with immediate blowout hazard.

### 6.2 Mandatory Step-Cut Box Geometry (IRC:82 Section 8.3)
Potholes cannot be patched along irregular jagged edges. Maintenance regulations mandate cutting rectangular vertical faces extended by $+15\text{ cm}$ outside the perimeter:
$$W_{\text{box}} = W_{\text{pothole}} + 15.0\text{ cm}$$
$$L_{\text{box}} = L_{\text{pothole}} + 15.0\text{ cm}$$
$$D_{\text{box}} = \max\left( D_{\text{fused}} + 1.0\text{ cm}, \ 4.0\text{ cm} \right)$$

* **Box Excavation Volume ($V_{\text{box}}$)**:
  $$V_{\text{box}} = \frac{W_{\text{box}} \times L_{\text{box}} \times D_{\text{box}}}{1000} \quad [\text{Liters}]$$

### 6.3 Dense Bituminous Cold-Mix Patch Mass & Bag Count
Standard compacted dense bituminous macadam (DBM) exhibits a compacted density of $\rho = 2.35\text{ kg/L}$. An allowance factor of $+18\%$ ($\kappa_{\text{shrink}} = 0.18$) is incorporated to account for loose uncompacted volume reduction under roller passes:
$$M_{\text{patch}} = V_{\text{box}} \times \rho_{\text{asphalt}} \times (1 + \kappa_{\text{shrink}}) = V_{\text{box}} \times 2.35 \times 1.18 \quad [\text{kg}]$$
* **Standard 25 kg Bag Count ($N_{\text{bags}}$)**:
  $$N_{\text{bags}} = \left\lceil \frac{M_{\text{patch}}}{25.0} \right\rceil$$

### 6.4 Tack Coat Emulsion Volume
A rapid-setting cationic bitumen emulsion (RS-1 per IS:8887) must be sprayed onto the excavated bed and vertical walls at a rate of $R_{\text{tack}} = 0.25\text{ to }0.30\text{ kg/m}^2$:
$$A_{\text{tack}} = \frac{W_{\text{box}} \cdot L_{\text{box}}}{10000} + \frac{2 \cdot (W_{\text{box}} + L_{\text{box}}) \cdot D_{\text{box}}}{10000} \quad [\text{m}^2]$$
$$M_{\text{tack}} = A_{\text{tack}} \times 0.30 \quad [\text{kg}]$$

### 6.5 Dynamic Suspension Travel Stroke Consumption
Standard passenger vehicle suspensions possess an available downward bump-stop travel envelope of $H_{\text{stop}} = 7.5\text{ cm}$. At vehicle velocity $v_x$ (km/h):
$$\Delta z_{\text{dyn}} = \min\left( 7.5\text{ cm}, \ D_{\text{fused}} \cdot \left[ \frac{v_x}{40.0} \right]^{0.42} \right)$$
$$\text{Stroke Consumption Pct} = \min\left( 100\%, \ \frac{\Delta z_{\text{dyn}}}{7.5} \times 100\% \right)$$
* $> 85\%$: Critical bottoming-out hazard (strut damage, loss of steering control).
* $55\text{--}85\%$: Moderate suspension fatigue and damper heating.
* $< 55\%$: Safe compliance range.

### 6.6 Tire Blowout Safe Speed Ceiling ($V_{\text{safe}}$)
Tire sidewall pinch and rim deformation scale non-linearly with impact kinetic energy:
$$V_{\text{safe}} = \max\left( 15.0\text{ km/h}, \ \min\left( 80.0\text{ km/h}, \ 40.0 \cdot \frac{4.2}{\max(D_{\text{fused}}, 1.0)} \right) \right)$$
* If $v_x > V_{\text{safe}} + 10\text{ km/h}$: **Severe Blowout Hazard** (sidewall shear, rim cracking).
* If $v_x > V_{\text{safe}}$: **Moderate Rim Impact Risk**.
* If $v_x \le V_{\text{safe}}$: Safe operational velocity.
