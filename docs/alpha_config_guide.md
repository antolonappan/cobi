# Alpha Configuration for Anisotropic Cosmic Birefringence

## Overview

The `alpha_config` parameter provides fine-grained control over how the anisotropic cosmic birefringence (CB) alpha field is generated for different simulation indices. This is particularly useful when you need:

1. **Varying alpha realizations** (default): Each simulation has a unique alpha map
2. **Constant alpha realizations**: All simulations in a range share the same alpha map
3. **Null alpha realizations**: No rotation applied (as if Acb=0)

## Configuration Structure

The `alpha_config` parameter accepts a dictionary with the following optional keys:

```python
alpha_config = {
    'alpha_vary_index': [start_idx, end_idx],   # Varying alpha (unique seed per idx)
    'alpha_cons_index': [start_idx, end_idx],   # Constant alpha (fixed seed)
    'null_alpha_index': [start_idx, end_idx]    # No rotation (alpha=0)
}
```

### Key Descriptions

- **`alpha_vary_index`**: Each simulation index in this range gets a unique seed, producing different alpha maps. This is the default behavior if no configuration is provided.

- **`alpha_cons_index`**: All simulation indices in this range use the same fixed seed (specifically, seed[0]), resulting in identical alpha maps across all realizations. The CMB realizations themselves remain different.

- **`null_alpha_index`**: Simulations in this range have no birefringence rotation applied, effectively setting Acb=0 for these indices only.

### Range Format

- Each value must be a list or tuple: `[start, end]`
- Ranges are **inclusive of start, exclusive of end** (Python convention: `start <= idx < end`)
- Ranges should not overlap
- Indices outside all defined ranges will use the default `alpha_vary_index` behavior

## Usage Example

### Basic Usage

```python
from cobi.simulation import LATsky

# Define your configuration
alpha_config = {
    'alpha_vary_index': [0, 400],      # Indices 0-399: varying alpha
    'alpha_cons_index': [400, 500],    # Indices 400-499: constant alpha
    'null_alpha_index': [500, 600]     # Indices 500-599: no rotation
}

# Create LATsky instance
lat = LATsky(
    libdir="/path/to/library",
    nside=512,
    cb_model='aniso',          # REQUIRED: Only works with 'aniso' model
    Acb=1e-6,
    lensing=True,
    alpha_config=alpha_config,  # Pass your configuration
    verbose=True
)

# Generate simulations
qu_varying = lat.cmb.get_cb_lensed_QU(10)    # Uses varying alpha
qu_constant = lat.cmb.get_cb_lensed_QU(410)  # Uses constant alpha
qu_null = lat.cmb.get_cb_lensed_QU(510)      # No rotation
```

### Use Cases

#### 1. Testing Statistical Consistency

```python
# Compare results with same alpha but different CMB realizations
config = {
    'alpha_cons_index': [0, 100],   # 100 sims with same alpha
}

lat = LATsky(..., alpha_config=config)

# All these share the same alpha field:
for idx in range(100):
    qu = lat.cmb.get_cb_lensed_QU(idx)
    # Process qu...
```

#### 2. Creating Null Tests

```python
# Generate some sims with CB and some without for comparison
config = {
    'alpha_vary_index': [0, 500],    # 500 sims with CB
    'null_alpha_index': [500, 1000]  # 500 sims without CB (null test)
}

lat = LATsky(..., alpha_config=config)
```

#### 3. Mixed Configuration

```python
# A realistic analysis configuration
config = {
    'alpha_vary_index': [0, 400],      # Main sims: varying CB
    'alpha_cons_index': [400, 500],    # Fixed CB for consistency tests
    'null_alpha_index': [500, 600]     # Null simulations for bias estimation
}
```

## Important Notes

### Model Requirement

⚠️ **The `alpha_config` parameter ONLY works with `cb_model='aniso'`**

If you use `cb_model='iso'` or `cb_model='iso_td'`, the `alpha_config` will be ignored with a warning.

### File Naming

The generated files maintain the same naming convention:
- Varying/Constant: `sims_nside{nside}_{idx:03d}.fits`
- Null: Same as above (just no rotation applied during generation)

The alpha maps are stored separately:
- `alpha_nside{nside}_{idx:03d}.fits`

For constant alpha range, all indices will reference/generate the same alpha map using seed[0].

### Performance Considerations

- **Constant alpha mode**: Saves computation time and disk space since the same alpha map is reused
- **Null alpha mode**: Slightly faster than regular simulations (no rotation computation)
- **Varying alpha mode**: Standard performance (default behavior)

## Validation

The configuration is validated when the `CMB` class is initialized:

```python
# These will raise ValueError:

# Invalid key
alpha_config = {'invalid_key': [0, 100]}  # ❌

# Invalid range format
alpha_config = {'alpha_vary_index': [0]}  # ❌ needs two values

# Invalid range values
alpha_config = {'alpha_vary_index': [100, 50]}  # ❌ start >= end
```

## Implementation Details

### Under the Hood

The implementation modifies the `alpha_alm()` method in the `CMB` class:

1. **Varying mode**: Uses `seed[idx]` for each realization
2. **Constant mode**: Uses `seed[0]` for all realizations in range
3. **Null mode**: Returns zero alm (no rotation)

The rotation is controlled in the `get_aniso_real_lensed_QU()` and `get_aniso_gauss_lensed_QU()` methods.

## Example Workflow

```python
from cobi.simulation import LATsky

# Step 1: Define your simulation strategy
alpha_config = {
    'alpha_vary_index': [0, 400],
    'alpha_cons_index': [400, 500],
    'null_alpha_index': [500, 600]
}

# Step 2: Initialize
lat = LATsky(
    libdir="/scratch/simulations",
    nside=512,
    cb_model='aniso',
    Acb=1e-6,
    lensing=True,
    alpha_config=alpha_config
)

# Step 3: Generate simulations
for idx in range(600):
    # Automatically handles the right mode based on idx
    qu = lat.cmb.get_cb_lensed_QU(idx)
    
    # Save to sky simulation
    lat.SaveObsQUs(idx)

# Step 4: Analysis
# You can now analyze knowing:
# - [0, 400): Each has unique CB signal
# - [400, 500): All share same CB pattern (isolates CMB variance)
# - [500, 600): No CB signal (for bias/systematic tests)
```

## Troubleshooting

### Issue: Configuration not working

**Check:**
1. Is `cb_model='aniso'`? (Required!)
2. Is `Acb` non-zero?
3. Are index ranges properly formatted as `[start, end]`?

### Issue: Getting same maps for varying mode

**Possible causes:**
- Files already exist and are being read from disk
- Check if you're in the constant alpha range instead

### Issue: No rotation being applied

**Check:**
- Are you in the `null_alpha_index` range?
- Is `Acb=0` in your initialization?

## Additional Resources

- See `examples/test_alpha_config.py` for a complete working example
- Check the CMB class docstrings for more technical details
- Review the `alpha_alm()` method for implementation specifics
