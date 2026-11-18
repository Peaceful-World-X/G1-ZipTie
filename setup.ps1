$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtualenv creation failed' }
}
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if (-not (Test-Path -LiteralPath 'assets\vendor\mujoco_menagerie\unitree_g1\g1.xml')) {
    git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/vendor/mujoco_menagerie
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
    git -C assets/vendor/mujoco_menagerie sparse-checkout set unitree_g1
    git -C assets/vendor/mujoco_menagerie checkout 370c43965323c0944b433b202aac5058b5f907a9
    if ($LASTEXITCODE -ne 0) { throw 'Pinned model checkout failed' }
}
if (-not (Test-Path -LiteralPath 'assets\vendor\unitree_ros\robots\dexterous_hand_description\dex1_1\dex1_1.urdf')) {
    git clone --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git assets/vendor/unitree_ros
    if ($LASTEXITCODE -ne 0) { throw 'Dex1-1 download failed' }
    git -C assets/vendor/unitree_ros sparse-checkout set robots/dexterous_hand_description/dex1_1
    git -C assets/vendor/unitree_ros checkout 157ff950351e5f535e03c5d85a16b7d594b3dbcc
    if ($LASTEXITCODE -ne 0) { throw 'Pinned Dex1-1 checkout failed' }
}
& .venv\Scripts\python.exe .\src\build_sequence.py
if ($LASTEXITCODE -ne 0) { throw 'Full-task scene generation failed' }
Write-Host 'Ready. Double-click 启动仿真.bat'
