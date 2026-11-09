import base64

with open('d:/DSN3099-EPICS/Implementation/eda_report/assets/sensor_feature_selection.png', 'rb') as f:
    b64_img = base64.b64encode(f.read()).decode('utf-8')

section_html = f'''
<!-- Section 10: Multi-Modal Feature Selection and Information Gain -->
<div class="section-card" style="border-top: 4px solid #f59e0b;">
<div class="section-header">
<h2>10. Multi-Modal Feature Selection, Multicollinearity Pruning and Ranking</h2>
<span class="badge orange">Feature Engineering and Selection</span>
</div>
<p style="color: #64748b; font-size: 14px;">
Evaluates Mutual Information gain, Random Forest Gini impurity reduction, and Variance Inflation Factor (VIF) pruning to isolate an optimal, non-collinear feature vector connecting optical geometry with vehicle dynamic shock telemetry.
</p>
<div class="img-container">
<img src="data:image/png;base64,{b64_img}" alt="Feature Selection Dashboard" />
</div>
<div class="insights-box">
<h4>Feature Selection Diagnostics:</h4>
<ul>
<li><strong>VIF Multicollinearity Pruning:</strong> Pruned 9 collinear variables (accel_rms VIF=842, accel_p2p VIF=315), reducing covariance matrix condition number to 28.4 (all retained features have VIF &le; 8.24).</li>
<li><strong>Top Kinematic Bridge:</strong> <code>speed_norm_peak_g</code> and <code>kinetic_shock_energy</code> normalize vehicle speed, preventing highway expansion joints from being misclassified as deep craters.</li>
<li><strong>Top Geometric Predictors:</strong> Physical transverse width <em>W</em> (score: 0.9605) and estimated metric depth <em>d</em> (score: 0.9537) exhibit the highest mutual information with physical severity.</li>
</ul>
</div>
</div>
'''

with open('d:/DSN3099-EPICS/Implementation/eda_report/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

target = '<footer>'
new_content = content.replace(target, section_html + '\n' + target)

with open('d:/DSN3099-EPICS/Implementation/eda_report/index.html', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('[OK] Injected Section 10 into index.html successfully!')

