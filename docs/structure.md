# File system logic

## Data

Currently:
```./data/DATA_PRIVACY/datasetname + .npy```

New: de-prioritise data privacy in structure; go for flat format but more descriptive file names
The AML variant will be pulling the data from stored Datasets, but I think they're also amenable to folder structure.


# Results and derived quantities

This section is more aspirational right now.

## Primary results

We compute results from many experiments, these are recorded as trace files containing loss, gradients, weights, etc.
These are the primary data for the project. 

They live in
```./traces/```

## Derived quantities

The derived quantities are computed from these results files and typically aggregate over many primary result files.

They consist of quantities such as:
- estimated sigma
- distribution of delta values
- distribution of sensitivity values
- performance of model versus epsilon

Such files live in
```./derived/```

## Figures

Figures or tables. These rely on derived quantities. They will have dedicated scripts which can be isolated from more general functionality.
If a figure requires data which is too specific (to the figure) for the derived quantities folder, it can live in the figures folder.
For example, if a plot requires many datapoints and it would be helpful to precompute these values. However, generally we should assume that derived quantities will be reused and should live in the derived folder.

The resulting figures live in
```./figures/```
