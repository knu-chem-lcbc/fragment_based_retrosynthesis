## fragment_based_retrosynthesis

Datasets for fragment-based retrosynthesis planning
1. PD1_RD1.txt.bz2	
 - Single reactant reactions
2. PD2_RD2.txt.bz2	
 - Double reactant reactions
3. PC6_RC6.txt.bz2
 - Mixed cases

Example entry from PD2_RD2.txt:

**Tz Px rx dx E D L p b y v r h i o a t e**     *dx D L v r o a t e - Tz p y i o a e*

---- 

## brief explanation of dataset curation

1. We used the USPTO dataset recently mined by Lowe[^1]. It contains 1,002,970 single product atom-mapped reactions. For the sake of simplicity, we restricted ourselves to single product reactions in this work. If it is considered that Lowe's dataset consists of 1,088,170 reactions in total with no duplicates, it is understood that limiting ourselves to single product reactions are not a preference but an obligation.
 - Number of reactions to start with : 1,002,970.
2. We removed the reactions if the number of molecules at reactant side is greater than or equal to three.
 - The size of the dataset at this stage : 922,823.
3. We also removed the duplicated reactions generated due to new representation.
 - The size of the dataset at this stage : 786,219.
4. We cleaned internal twins, where product and reactant are identical on unimolecular reactions.
 - The size of the dataset at this stage : 780,471.

In this stage of the corpus; we have 186,206 unimolecular reactions, and 594,265 reactions where reactant side is composed of two molecules.

5. We then examined the distribution of the reactions w.r.t total length of product and reaction pairs. For instance, The following reaction : *[ C ix Wx I f u w o a e >> C I u w o a e - C ix Wx I f u w o a e ]* has total length 27.
We first set a maximum total length for a reaction as 100, and reduce the size of the dataset accordingly.
 - The size of the dataset at this stage : 429,889.
6. For NMT applications, injection property is something desirable. In cases where a product maps into different reactants at different reactions, it is better having an injection from product domain to reactant domain.  Therefore, we identified all such cases and applied a recipe to select only one of those reactions. We rioritized two reactants over a single reactant, and selected the one with minimum total lenght.
 - The size of the dataset at this stage : 352,546 (namely **Mixed.txt** dataset which contains single and double reactant reactions)
 
[^1]: Lowe, D.M.: Extraction of chemical structures and reactions from theliterature. PhD thesis, University of Cambridge (2012).
