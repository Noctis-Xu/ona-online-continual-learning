//ONA-TIP: support time, temporal alignment and read invariants.
#include <fenv.h>

void TIProjection_Test()
{
#if TEMPORAL_IMPLICATION_PROJECTION
    puts(">>ONA-TIP test start");
    long savedTime = currentTime;
    long halfLife = TEMPORAL_IMPLICATION_PROJECTION_HALF_LIFE;
    Concept source = {0};
    Implication original = {
        .term = Narsese_Term("<signal =/> outcome>"),
        .truth = {0.8, 0.9}, .truthTime = 100,
        .occurrenceTimeOffset = 7, .creationTime = 99,
        .sourceConcept = &source, .stamp = {{41}}
    };
    Implication once = Implication_Project(original, 100 + halfLife);
    Implication twice = Implication_Project(once, 100 + 2*halfLife);
    Implication direct = Implication_Project(original, 100 + 2*halfLife);
    assert(fabs(once.truth.confidence - 0.45) < 1e-12, "Projection must halve confidence");
    assert(Truth_Equal(&twice.truth, &direct.truth), "Forward projection must compose");
    assert(original.truth.confidence == 0.9 && original.truthTime == 100, "Read must preserve the support time");
    assert(once.truth.frequency == original.truth.frequency && once.occurrenceTimeOffset == 7, "Projection must preserve frequency and lag");
    assert(Stamp_Equal(&once.stamp, &original.stamp), "Projection must preserve stamp");

    currentTime = 100 + halfLife;
    Table table = {0};
    Table_AddAndRevise(&table, &original); //delayed insertion, not new evidence
    Implication read = Table_ReadAt(&table, 0, currentTime);
    assert(fabs(read.truth.confidence - 0.45) < 1e-12, "Delayed insertion must keep original time");
    assert(table.array[0].truthTime == 100, "Insertion must not refresh the support time");
    Table_AddAndRevise(&table, &original);
    read = Table_ReadAt(&table, 0, currentTime);
    assert(fabs(read.truth.confidence - 0.45) < 1e-12, "Reprocessing the same stamp must not add evidence");
    assert(table.array[0].truthTime == original.truthTime && table.array[0].truth.confidence == original.truth.confidence, "Choice must preserve original support, not write the comparison copy");
    Implication past = Table_ReadAt(&table, 0, original.truthTime);
    assert(Truth_Equal(&past.truth, &original.truth), "Choice must preserve the support-time view");
    Implication weaker = original;
    weaker.truthTime = currentTime;
    weaker.truth.confidence = 0.3;
    Implication winner = Inference_ImplicationRevision(&original, &weaker, currentTime);
    assert(winner.truthTime == original.truthTime && winner.truth.confidence == original.truth.confidence, "Choice must preserve the older winner's support time");
    winner = Inference_ImplicationRevision(&weaker, &original, currentTime);
    assert(winner.truthTime == original.truthTime && winner.truth.confidence == original.truth.confidence, "Choice must preserve either winning operand");

    Implication fresh = original;
    fresh.truth = (Truth){0.0, 0.9};
    fresh.truthTime = currentTime;
    fresh.stamp = (Stamp){{42}};
    Table_AddAndRevise(&table, &fresh);
    assert(fabs(table.array[0].truth.frequency - (0.8 / 12.0)) < 1e-12, "Revision must align old and new evidence");
    assert(table.array[0].truthTime == currentTime, "Revised truth must have the aligned time");

    table = (Table){.itemsAmount = 2};
    table.array[0] = original;
    table.array[1] = fresh;
    table.array[1].truth = (Truth){1.0, 0.3};
    Table_RankAt(&table, currentTime);
    assert(table.array[0].stamp.evidentialBase[0] == 42, "Ranking must use projected expectation");
    Implication tiny = Implication_Project(original, 100 + 2000*halfLife);
    assert(tiny.truth.confidence == 0.0, "Old support may underflow to ignorance");
    Implication tinyOther = tiny;
    tinyOther.stamp = (Stamp){{43}};
    tinyOther.occurrenceTimeOffset = 11;
    feclearexcept(FE_ALL_EXCEPT);
    Implication zero = Inference_ImplicationRevision(&tiny, &tinyOther, tiny.truthTime);
    assert(!fetestexcept(FE_INVALID | FE_DIVBYZERO), "TI zero support must bypass undefined interval arithmetic");
    assert(zero.truth.confidence == 0.0 && zero.occurrenceTimeOffset == tinyOther.occurrenceTimeOffset, "Zero support must keep a finite interval without adding confidence");

    Implication supported = tinyOther;
    supported.truth.confidence = 0.5;
    Implication oneSupported = Inference_ImplicationRevision(&tiny, &supported, tiny.truthTime);
    assert(oneSupported.occurrenceTimeOffset == supported.occurrenceTimeOffset && oneSupported.truth.confidence == 0.5, "One supported operand must use normal weighted revision");

    //ONA-TIP must preserve upstream behavior for ordinary implications.
    tiny.term = Narsese_Term("<signal ==> outcome>");
    tinyOther.term = tiny.term;
    Implication ordinaryZero = Inference_ImplicationRevision(&tiny, &tinyOther, tiny.truthTime);
    assert(isnan(ordinaryZero.occurrenceTimeOffset) && ordinaryZero.truth.confidence == 0.0, "Ordinary zero-support revision must retain upstream behavior");
    feclearexcept(FE_ALL_EXCEPT);
    currentTime = savedTime;
    puts("<<ONA-TIP test successful");
#endif
}
